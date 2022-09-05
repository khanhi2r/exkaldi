import os
import argparse

CONDA_DIR = "/home/khanh/workspace/miniconda3"
KALDI_ENV = "kaldi"
EXKALDI_ENV = "exkaldi"
KALDI_ROOT = "/home/khanh/workspace/projects/kaldi"

DATA_DIR = "librispeech_dummy"

def import_exkaldi():
    import os

    # add lib path
    os.environ["LD_LIBRARY_PATH"] = ";".join([
        os.path.join(CONDA_DIR, "envs", KALDI_ENV, "lib"),
        os.path.join(CONDA_DIR, "envs", EXKALDI_ENV, "lib"),
    ])

    import exkaldi
    exkaldi.info.reset_kaldi_root(KALDI_ROOT)

    return exkaldi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--low-level", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    stage = args.stage
    low_level = args.low_level

    exkaldi = import_exkaldi()

    if stage <= 2: # MFCC
        print("### MFCC ###")
        scp_path = os.path.join(DATA_DIR, "train", "wav.scp")
        feat = exkaldi.compute_mfcc(scp_path, name="mfcc")

        spk2utt_path = os.path.join(DATA_DIR, "train", "spk2utt")
        cmvn = exkaldi.compute_cmvn_stats(feat, spk2utt=spk2utt_path, name="cmvn")

        utt2spk_path = os.path.join(DATA_DIR, "train", "utt2spk")
        feat = exkaldi.use_cmvn(feat, cmvn, utt2spk=utt2spk_path)

        feat_path = os.path.join(DATA_DIR, "exp", "train_mfcc_cmvn.ark")
        exkaldi.utils.make_dependent_dirs(path=feat_path, pathIsFile=True)
        feat_index = feat.save(feat_path, returnIndexTable=True)
    
    if stage <= 3: # LEXICON
        print("### LEXICON ###")
        lexicon_path = os.path.join(DATA_DIR, "pronunciation.txt")
        sil_words={
            "<SIL>":"<SIL>",
            "<SPN>":"<SPN>",
        }
        unk_symbol={
            "<UNK>":"<SPN>",
        }
        optional_sil_phone = "<SIL>"

        lexicons = exkaldi.decode.graph.lexicon_bank(
            lexicon_path,
            sil_words,
            unk_symbol, 
            optional_sil_phone, 
            positionDependent=False,
            shareSilPdf=False,
        )


        words_path = os.path.join(DATA_DIR, "exp", "words.txt")
        exkaldi.utils.make_dependent_dirs(path=words_path, pathIsFile=True)
        lexicons.dump_dict(name="words", fileName=words_path, dumpInt=False)

        L_path = os.path.join(DATA_DIR,"exp","L.fst")
        exkaldi.decode.graph.make_L(lexicons, outFile=L_path, useDisambigLexicon=False)

        L_disambig_path = os.path.join(DATA_DIR,"exp","L_disambig.fst")
        exkaldi.decode.graph.make_L(lexicons, outFile=L_disambig_path, useDisambigLexicon=True)

        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons.save(lex_path)

    if stage <= 4: # LANGUAGE MODEL
        print("### LANGUAGE MODEL ###")
        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons = exkaldi.load_lex(lex_path)

        text_path = os.path.join(DATA_DIR, "train", "text")
        trans = exkaldi.load_transcription(text_path)

        new_text_path = os.path.join(DATA_DIR, "exp", "train_lm_text")
        trans.save(fileName=new_text_path, discardUttID=True)

        arpa_path = os.path.join(DATA_DIR, "exp", "2-gram.arpa")
        exkaldi.lm.train_ngrams_kenlm(lexicons, order=2, text=trans, outFile=arpa_path, config={"-S":"20%"})

        G_path = os.path.join(DATA_DIR, "exp", "G.fst")
        exkaldi.decode.graph.make_G(lexicons, arpa_path, outFile=G_path, order=2)
    
    if stage <= 5: # MONO HMM GMM
        print("### MONO HMM GMM ###")
        # prepare int-id format transcription
        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons = exkaldi.load_lex(lex_path)

        topo_path = os.path.join(DATA_DIR, "exp", "topo")
        exkaldi.hmm.make_topology(lexicons, outFile=topo_path, numNonsilStates=3, numSilStates=3)

        feat_path = os.path.join(DATA_DIR, "exp", "train_mfcc_cmvn.ark")
        feat = exkaldi.load_feat(feat_path, name="mfcc")

        feat = feat.add_delta(order=2)

        model = exkaldi.hmm.MonophoneHMM(lexicons=lexicons, name="mono")
        model.initialize(feat=feat, topoFile=topo_path)

        trans_path = os.path.join(DATA_DIR, "train", "text")
        model_dir = os.path.join(DATA_DIR, "exp", "train_mono")
        
        L_path = os.path.join(DATA_DIR,"exp","L.fst")

        
        if low_level:
            # low_level
            trans = exkaldi.hmm.transcription_to_int(trans_path, lexicons)

            # compile the train graph
            graph_path = os.path.join(model_dir, "train_graph")
            model.compile_train_graph(tree=model.tree, transcription=trans, LFile=L_path, outFile=graph_path)

            num_iterations = 10
            for i in range(num_iterations):
                print(f"pass {i}")
                print("\taligning...")
                if i == 0:
                    # align acoustic feature averagely in order to start the first train step.
                    ali = model.align_equally(feat, graph_path)
                else:
                    # align acoustic feature with Vertibi algorithm
                    ali = model.align(feat=feat, trainGraphFile=graph_path)

                print("\taccumulating stats...")
                # use alignment to accumulate the statistics in order to update the parameters of model
                stats_path = os.path.join(model_dir, "stats.acc")
                model.accumulate_stats(feat=feat, ali=ali, outFile=stats_path)

                print("\tupdating model...")
                # use these statistics to update model parameters.
                target_gaussians = min(500, model.info.gaussians + 8) # increase number of gaussians every pass
                model.update(stats_path, numgauss=target_gaussians)

                # save model manually
                model_path = os.path.join(model_dir, "final.mdl")
                model.save(model_path)
                tree_path = os.path.join(model_dir, "tree")
                model.tree.save(tree_path)
                ali_path = os.path.join(model_dir, "final.ali")
                ali.save(ali_path)

        else:
            # HIGH_LEVEL_API
            ali = model.train(feat, trans_path, L_path, tempDir=model_dir, numIters=10, maxIterInc=8, totgauss=500)

        

    if stage <= 6: # DECISION TREE
        print("### DECISION TREE ###")
        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons = exkaldi.load_lex(lex_path)
        tree = exkaldi.hmm.DecisionTree(lexicons=lexicons, contextWidth=3, centralPosition=1)

        feat_path = os.path.join(DATA_DIR, "exp", "train_mfcc_cmvn.ark")
        feat = exkaldi.load_feat(feat_path)
        feat = feat.add_delta(order=2)

        hmm_path = os.path.join(DATA_DIR, "exp", "train_mono", "final.mdl")

        ali_path = os.path.join(DATA_DIR, "exp", "train_mono", "final.ali")
        ali = exkaldi.load_index_table(ali_path, useSuffix="ark")

        model_dir = os.path.join(DATA_DIR, "exp", "train_delta")
        topo_path = os.path.join(DATA_DIR, "exp", "topo")

        if low_level:
            # low_level
            # accumulate statistics data
            tree_stats_path = os.path.join(model_dir, "tree_stats.acc")
            tree.accumulate_stats(feat, hmm_path, ali, outFile=tree_stats_path)

            # cluster phones and compile questions.
            questions_path = os.path.join(model_dir, "questions.qst")
            tree.compile_questions(tree_stats_path, topo_path, outFile=questions_path)

            # build tree
            target_leaves = 300
            tree.build(tree_stats_path, questions_path, topo_path, numLeaves=target_leaves)

            tree_path = os.path.join(model_dir, "tree")
            tree.save(tree_path)
        else:
            # HIGH_LEVEL_API
            tree = exkaldi.hmm.DecisionTree(lexicons=lexicons,contextWidth=3,centralPosition=1)
            tree.train(feat=feat, hmm=hmm_path, ali=ali, topoFile=topo_path, numLeaves=300, tempDir=model_dir)
            os.rename(os.path.join(model_dir, "treeStats.acc"), os.path.join(model_dir, "tree_stats.acc"))

    if stage <= 7: # TRI HMM GMM DELTA
        print("### TRI HMM GMM DELTA ###")

        tree_path = os.path.join(DATA_DIR, "exp", "train_delta", "tree")
        tree_stats_path = os.path.join(DATA_DIR, "exp", "train_delta", "tree_stats.acc")
        topo_path = os.path.join(DATA_DIR, "exp", "topo")

        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons = exkaldi.load_lex(lex_path)

        feat_path = os.path.join(DATA_DIR, "exp", "train_mfcc_cmvn.ark")
        feat = exkaldi.load_feat(feat_path)
        feat = feat.add_delta(order=2)

        L_path = os.path.join(DATA_DIR, "exp", "L.fst")
        model_dir = os.path.join(DATA_DIR, "exp", "train_delta")
        trans_path = os.path.join(DATA_DIR, "train", "text")

        
        if low_level:
            # low_level
            ali_path = os.path.join(DATA_DIR, "exp", "train_mono", "final.ali")
            mono_path= os.path.join(DATA_DIR, "exp", "train_mono", "final.mdl")
            ali = exkaldi.load_ali(ali_path)

            model = exkaldi.hmm.TriphoneHMM()
            model.initialize(tree=tree_path, treeStatsFile=tree_stats_path, topoFile=topo_path)

            new_ali = exkaldi.hmm.convert_alignment(
                ali=ali, 
                originHmm=mono_path, 
                targetHmm=model, 
                tree=tree_path
            )

            
            
            trans = exkaldi.hmm.transcription_to_int(trans_path, lexicons)
            
            #int_trans_path = os.path.join(DATA_DIR, "exp", "text.int")
            #trans.save(int_trans_path)

            # compile new train graph
            graph_path = os.path.join(model_dir, "train_graph")
            model.compile_train_graph(tree=model.tree, transcription=trans, LFile=L_path, outFile=graph_path, lexicons=lexicons) # try to replace tree_path by model.tree


            num_iterations = 10
            for i in range(num_iterations):
                print(f"pass {i}")
                print("\taligning...")
                # align acoustic feature
                ali = model.align(feat, graph_path, lexicons=lexicons)
                
                print("\taccumulating stats...")
                # accumulate statistics
                stats_path = os.path.join(model_dir, "stats.acc")
                model.accumulate_stats(feat=feat, ali=ali, outFile=stats_path)

                print("\tupdating model...")
                # update HMM GMM parameters
                target_gaussians = min(1500, model.info.gaussians + 8) # increase number of gaussians every pass
                model.update(stats_path, target_gaussians)
            
            # save model manually
            model_path = os.path.join(model_dir, "final.mdl")
            model.save(model_path)
            tree_path = os.path.join(model_dir, "tree")
            model.tree.save(tree_path)
            ali_path = os.path.join(model_dir, "final.ali")
            ali.save(ali_path)

        else:
            # HIGH_LEVEL_API
            model = exkaldi.hmm.TriphoneHMM(lexicons=lexicons)
            model.initialize(tree=tree_path, treeStatsFile=tree_stats_path, topoFile=topo_path)

            ali_index = model.train(
                feat=feat,
                transcription=trans_path,
                LFile=L_path,
                tree=tree_path,
                tempDir=model_dir, 
                numIters=10,
                maxIterInc=8,
                totgauss=1500,
            )

    if stage <= 8: # MAKE HCLG
        print("### MAKE HCLG ###")
        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons = exkaldi.load_lex(lex_path)

        graph_dir = os.path.join(DATA_DIR, "exp", "train_delta", "graph")
        hmm_path = os.path.join(DATA_DIR, "exp", "train_delta", "final.mdl")
        tree_path = os.path.join(DATA_DIR, "exp", "train_delta", "tree")
        L_path = os.path.join(DATA_DIR, "exp", "L_disambig.fst")
        G_path = os.path.join(DATA_DIR, "exp", "G.fst")

        if low_level:
            # LOW_LEVEL_API
            
            LG_path = os.path.join(graph_dir, "LG.fst")
            exkaldi.decode.graph.compose_LG(L_path, G_path, outFile=LG_path)

            
            CLG_path = os.path.join(graph_dir, "CLG.fst")
            _, i_label_path = exkaldi.decode.graph.compose_CLG(lexicons, tree_path, LG_path, outFile=CLG_path)

            
            HCLG_path = os.path.join(graph_dir, "HCLG.fst")
            exkaldi.decode.graph.compose_HCLG(hmm_path, tree_path, CLG_path, i_label_path, outFile=HCLG_path)
        else:
            # HIGH_LEVEL_API
            exkaldi.decode.graph.make_graph(lexicons, hmm_path, tree_path, tempDir=graph_dir, useLFile=L_path, useGFile=G_path)

    if stage <= 9: # DECODE HMM GMM
       print("### DECODE HMM GMM ###")
        lex_path = os.path.join(DATA_DIR, "exp", "lexicons.lex")
        lexicons = exkaldi.load_lex(lex_path)

        scp_path = os.path.join(DATA_DIR, "test", "wav.scp")
        utt2spk_path = os.path.join(DATA_DIR, "test", "utt2spk")
        spk2utt_path = os.path.join(DATA_DIR, "test", "spk2utt")

        feat = exkaldi.compute_mfcc(scp_path, name="mfcc")
        cmvn = exkaldi.compute_cmvn_stats(feat, spk2utt=spk2utt_path, name="cmvn")
        feat = exkaldi.use_cmvn(feat, cmvn, utt2spk=utt2spk_path)

        feat_path = os.path.join(DATA_DIR, "exp", "test_mfcc_cmvn.ark")
        feat.save(feat_path)
        feat = feat.add_delta(order=2)

        HCLG_path = os.path.join(DATA_DIR, "exp", "train_delta", "graph", "HCLG.fst")

        hmm_path = os.path.join(DATA_DIR, "exp", "train_delta", "final.mdl")

        lat = exkaldi.decode.wfst.gmm_decode(feat, hmm_path, HCLG_path, symbolTable=lexicons("words"))

        decode_dir = os.path.join(DATA_DIR, "exp", "train_delta", "decode_test")
        lat_path = os.path.join(decode_dir, "test.lat")
        lat.save(lat_path)

    if stage <= 10: # SCORING
        print("### SCORING ###")
        lat_path = os.path.join(dataDir, "exp", "train_delta", "decode_test", "test.lat")
        lat = exkaldi.decode.wfst.load_lat(lat_path)

        wordsFile = os.path.join(dataDir, "exp", "words.txt")

        hmmFile = os.path.join(dataDir, "exp", "train_delta", "final.mdl")
    
        result = lat.get_1best(symbolTable=wordsFile, hmm=hmmFile, lmwt=1, acwt=0.5)

        textResult = exkaldi.hmm.transcription_from_int(result, wordsFile)

        lexFile = os.path.join(dataDir, "exp", "lexicons.lex")

        lexicons = exkaldi.load_lex(lexFile)

        

















    

