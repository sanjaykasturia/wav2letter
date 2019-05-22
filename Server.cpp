/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "decoder/Decoder.hpp"
#include "decoder/KenLM.hpp"
#include "decoder/Trie.hpp"
#include "module/module.h"
#include "runtime/Data.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"


#include "server_http.hpp"
// Added for the json-example
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// Added for the default_resource example
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>
using namespace std;
// Added for the json-example:
using namespace boost::property_tree;

using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

using namespace w2l;

int main(int argc, char** argv) {
	std::string exec(argv[0]);
	std::vector<std::string> argvs;
	for (int i = 0; i < argc; i++) {
		argvs.emplace_back(argv[i]);
	}
	gflags::SetUsageMessage(
			"Usage: \n " + exec + " [data_path] [dataset_name] [flags]");
	if (argc <= 1) {
		std:: cerr << gflags::ProgramUsage();
	}


	HttpServer server;
	server.config.port = 8080;
	/* ===================== Parse Options ===================== */
	std:: cout << "Parsing command line flags";
	gflags::ParseCommandLineFlags(&argc, &argv, false);
	auto flagsfile = FLAGS_flagsfile;
	if (!flagsfile.empty()) {
		std:: cout << "Reading flags from file " << flagsfile;
		gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
	}

	/* ===================== Create Network ===================== */
	if (!(FLAGS_am.empty() ^ FLAGS_emission_dir.empty())) {
		std:: cerr
			<< "One and only one of flag -am and -emission_dir should be set.";
	}
	EmissionSet emissionSet;

	/* Using acoustic model */
	std::shared_ptr<fl::Module> network;
	std::shared_ptr<SequenceCriterion> criterion;
	if (!FLAGS_am.empty()) {
		std::unordered_map<std::string, std::string> cfg;
		std:: cout << "[Network] Reading acoustic model from " << FLAGS_am;

		W2lSerializer::load(FLAGS_am, cfg, network, criterion);
		network->eval();
		std:: cout << "[Network] " << network->prettyString();
		if (criterion) {
			criterion->eval();
			std:: cout << "[Network] " << criterion->prettyString();
		}
		std:: cout << "[Network] Number of params: " << numTotalParams(network);

		auto flags = cfg.find(kGflags);
		if (flags == cfg.end()) {
			std:: cerr << "[Network] Invalid config loaded from " << FLAGS_am;
		}
		std:: cout << "[Network] Updating flags from config file: " << FLAGS_am;
		gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
	}
	/* Using existing emissions */
	else {
		std::string cleanedTestPath = cleanFilepath(FLAGS_test);
		std::string loadPath =
			pathsConcat(FLAGS_emission_dir, cleanedTestPath + ".bin");
		std:: cout << "[Serialization] Loading file: " << loadPath;
		W2lSerializer::load(loadPath, emissionSet);
		gflags::ReadFlagsFromString(emissionSet.gflags, gflags::GetArgv0(), true);
	}

	// override with user-specified flags
	gflags::ParseCommandLineFlags(&argc, &argv, false);
	if (!flagsfile.empty()) {
		gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
	}

	std:: cout << "Gflags after parsing \n" << serializeGflags("; ");

	/* ===================== Create Dictionary ===================== */

	auto tokenDict = createTokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
	int numClasses = tokenDict.indexSize();
	std:: cout << "Number of classes (network): " << numClasses;

	auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
	auto wordDict = createWordDict(lexicon);
	std:: cout << "Number of words: " << wordDict.indexSize();

	DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

	/* ===================== Create Dataset ===================== */

	if (FLAGS_criterion == kAsgCriterion) {
		emissionSet.transition = afToVector<float>(criterion->param(0).array());
	}

	int nSample = emissionSet.emissions.size();
	nSample = FLAGS_maxload > 0 ? std::min(nSample, FLAGS_maxload) : nSample;
	int nSamplePerThread =
		std::ceil(nSample / static_cast<float>(FLAGS_nthread_decoder));
	std:: cout << "[Dataset] Number of samples per thread: " << nSamplePerThread;

	/* ===================== Decode ===================== */


	// Prepare criterion
	ModelType modelType = ModelType::ASG;
	if (FLAGS_criterion == kCtcCriterion) {
		modelType = ModelType::CTC;
	} else if (FLAGS_criterion != kAsgCriterion) {
		std:: cerr << "[Decoder] Invalid model type: " << FLAGS_criterion;
	}

	const auto& transition = emissionSet.transition;

	// Prepare decoder options
	DecoderOptions decoderOpt(
			FLAGS_beamsize,
			static_cast<float>(FLAGS_beamscore),
			static_cast<float>(FLAGS_lmweight),
			static_cast<float>(FLAGS_wordscore),
			static_cast<float>(FLAGS_unkweight),
			FLAGS_logadd,
			static_cast<float>(FLAGS_silweight),
			modelType);

	// Build Language Model
	std::shared_ptr<LM> lm;
	if (FLAGS_lmtype == "kenlm") {
		lm = std::make_shared<KenLM>(FLAGS_lm);
		if (!lm) {
			std:: cerr << "[LM constructing] Failed to load LM: " << FLAGS_lm;
		}
	} else {
		std:: cerr << "[LM constructing] Invalid LM Type: " << FLAGS_lmtype;
	}
	std:: cout << "[Decoder] LM constructed.\n";

	// Build Trie
	if (std::strlen(kSilToken) != 1) {
		std:: cerr << "[Decoder] Invalid unknown_symbol: " << kSilToken;
	}
	if (std::strlen(kBlankToken) != 1) {
		std:: cerr << "[Decoder] Invalid unknown_symbol: " << kBlankToken;
	}
	int silIdx = tokenDict.getIndex(kSilToken);
	int blankIdx =
		FLAGS_criterion == kCtcCriterion ? tokenDict.getIndex(kBlankToken) : -1;
	int unkIdx = lm->index(kUnkToken);
	std::shared_ptr<Trie> trie =
		std::make_shared<Trie>(tokenDict.indexSize(), silIdx);
	auto start_state = lm->start(false);

	for (auto& it : lexicon) {
		std::string word = it.first;
		int lmIdx = lm->index(word);
		float score;
		auto dummyState = lm->score(start_state, lmIdx, score);
		for (auto& tokens : it.second) {
			auto tokensTensor = tokens2Tensor(tokens, tokenDict);
			trie->insert(
					tokensTensor,
					std::make_shared<TrieLabel>(lmIdx, wordDict.getIndex(word)),
					score);
		}
	}
	std:: cout << "[Decoder] Trie planted.\n";

	// Smearing
	SmearingMode smear_mode = SmearingMode::NONE;
	if (FLAGS_smearing == "logadd") {
		smear_mode = SmearingMode::LOGADD;
	} else if (FLAGS_smearing == "max") {
		smear_mode = SmearingMode::MAX;
	} else if (FLAGS_smearing != "none") {
		std:: cerr << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
	}
	trie->smear(smear_mode);
	std:: cout << "[Decoder] Trie smeared.\n";

	// Decoding



  auto runDecoder = [&](string inputfile) {
    try {
			// Build Decoder
			std::shared_ptr<TrieLabel> unk =
				std::make_shared<TrieLabel>(unkIdx, wordDict.getIndex(kUnkToken));
			Decoder decoder(decoderOpt, trie, lm, silIdx, blankIdx, unk, transition);

			// Get data and run decoder

			auto audiopath = inputfile; // 16-bit Signed Integer PCM
			auto input = speech ::loadSound<float>(audiopath.c_str());
			std:: cout << audiopath << std:: endl;
			// Double
			auto feat = featurizeInput(input);
			auto rawEmission = network->forward({fl::input(af::array( feat.inputDims , feat.input.data()))}).front();
			int N = rawEmission.dims(0);
			int T = rawEmission.dims(1);

			auto emission = afToVector<float>(rawEmission);

			std::vector<float> score;
			std::vector<std::vector<int>> wordPredictions;
			std::vector<std::vector<int>> letterPredictions;

			std::tie(score, wordPredictions, letterPredictions) =
				decoder.decode(emission.data(), T, N);

			// Cleanup predictions
			auto wordPrediction = wordPredictions[0];
			auto letterPrediction = letterPredictions[0];
			if (FLAGS_criterion == kCtcCriterion ||
					FLAGS_criterion == kAsgCriterion) {
				uniq(letterPrediction);
			}
			if (FLAGS_criterion == kCtcCriterion) {
				letterPrediction.erase(
						std::remove(
							letterPrediction.begin(), letterPrediction.end(), blankIdx),
						letterPrediction.end());
			}
			validateTokens(wordPrediction, wordDict.getIndex(kUnkToken));
			validateTokens(letterPrediction, -1);

			// Update meters & print out predictions

			auto wordPredictionStr = tensor2words(wordPrediction, wordDict);
			
			//TEST.cpp code for predicting letters:
			auto viterbiPath =
        afToVector<int>(criterion->viterbiPath(rawEmission.array()));
			if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kAsgCriterion) {
				uniq(viterbiPath);
			}
			if (FLAGS_criterion == kCtcCriterion) {
				auto blankidx = tokenDict.getIndex(kBlankToken);
				viterbiPath.erase(
						std::remove(viterbiPath.begin(), viterbiPath.end(), blankidx),
						viterbiPath.end());
			}
			remapLabels(viterbiPath, tokenDict);
			auto letterPredictionStr = tensor2letters(viterbiPath, tokenDict);
			
			std::cout << "letterPredictionStr: " << letterPredictionStr << std:: endl;
			std::cout << "wordPredictionStr: "<< wordPredictionStr << std:: endl;
			return letterPredictionStr + "\n" + wordPredictionStr;
		} catch (const std::exception& exc) {
			std:: cerr << "Exception : " << exc.what();
		}
  };

	server.resource["^/transcribe$"]["POST"] = [runDecoder](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    try {
      ptree pt;
      read_json(request->content, pt);

      auto name=pt.get<string>("inputfile");
      auto res = runDecoder(name);
      response->write(res);
    }
    catch(const exception &e) {
      response->write(SimpleWeb::StatusCode::client_error_bad_request, e.what());
    }
	};


      std:: cout << "Staring Server " << std:: endl;
			server.start();


	return 0;
}
