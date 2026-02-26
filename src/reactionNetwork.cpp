#include <algorithm>
#include <cassert>
#include <charconv>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "../include/constants.hpp"
#include "../include/readcsv.hpp"
#include "../include/reactionNetwork.hpp"

static std::string strip(std::string s) {
	size_t start = s.find_first_not_of(" \t\n\r");
	size_t end = s.find_last_not_of(" \t\n\r");
	return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

static bool is_stoiable(std::string s) {
	s = strip(std::move(s));
	long v;
	auto res = std::from_chars(s.data(), s.data() + s.size(), v);
	return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

ReactionNetwork::ReactionNetwork() {
	speciesData_.assign(config::species + 1, 0.0);
	speciesData_[config::species] = 1.0; // dummy
}

auto ReactionNetwork::RhsTerm::operator<=>(const RhsTerm& other) const {
	if (reactant2 == other.reactant2) {
		if (reactant1 == other.reactant1) {
			return add_to <=> other.add_to;
		}
		return reactant1 <=> other.reactant1;
	}
	return reactant2 <=> other.reactant2;
}

auto ReactionNetwork::JacTerm::operator<=>(const JacTerm& other) const {
	if (reactant == other.reactant) {
		return rateConstant <=> other.rateConstant;
	}
	return reactant <=> other.reactant;
}

void ReactionNetwork::build() {
	// reset
	termIndex.clear();
	rhsTerms.clear();

	jacNonZeros = 0;
	jacTerms.clear();
	jacIdxPointer.clear();
	jacIdxValue.clear();

	// dummy species
	if ((int)speciesData_.size() != config::species + 1) {
		speciesData_.assign(config::species + 1, 0.0);
	}
	speciesData_[config::species] = 1.0;

	auto csv_data = read_csv(std::string(config::reactNetworkFile));
	std::vector<std::map<std::string, std::string>> dataDict;
	dataDict.reserve(csv_data.size());
	for (size_t i = 1; i < csv_data.size(); ++i) {
		std::map<std::string, std::string> row;
		for (size_t j = 0; j < std::min(csv_data[i].size(), csv_data[0].size()); ++j) {
			row[strip(csv_data[0][j])] = strip(csv_data[i][j]);
		}
		dataDict.push_back(std::move(row));
	}

	// build kind -> index
	for (const auto& mp : dataDict) {
		std::string kind = mp.at("kind");
		termIndex[kind] = 0;
	}
	int idx = 0;
	for (auto& [key, val] : termIndex) {
		val = idx;
		++idx;
	}

	// RHS terms
	{
		std::map<std::tuple<int, int, int, int>, int> ode; // (add_to, init, entering, kind) -> duplicacy
		auto addTerm = [&](int add_to, int init, int entering, int kind, int duplicate_count) {
			std::tuple<int, int, int, int> key = std::make_tuple(add_to, init, entering, kind);
			auto it = ode.find(key);
			if (it != ode.end()) it->second += duplicate_count;
			else ode.emplace(key, duplicate_count);
		};

		for (const auto& mp : dataDict) {
			int init = std::stoi(mp.at("init_assem_id"));
			int entering = is_stoiable(mp.at("entering_assem_id")) ? std::stoi(mp.at("entering_assem_id")) : config::species;
			int product = std::stoi(mp.at("product_assem_id"));
			int leaving = is_stoiable(mp.at("leaving_assem_id")) ? std::stoi(mp.at("leaving_assem_id")) : config::species;
			int kind = termIndex.at(mp.at("kind"));
			int duplicacy = std::stoi(mp.at("duplicate_count"));

			addTerm(init, init, entering, kind, -duplicacy);
			if (entering != config::species) {
				addTerm(entering, init, entering, kind, -duplicacy);
			}
			addTerm(product, init, entering, kind, duplicacy);
			if (leaving != config::species) {
				addTerm(leaving, init, entering, kind, duplicacy);
			}
		}

		rhsTerms.reserve(ode.size());
		for (const auto& [key, duplicacy] : ode) {
			RhsTerm t;
			t.add_to = std::get<0>(key);
			t.reactant1 = std::get<1>(key);
			t.reactant2 = std::get<2>(key);
			t.rateConstant = std::get<3>(key);
			t.duplicacy = duplicacy;
			rhsTerms.push_back(t);
		}
		std::sort(rhsTerms.begin(), rhsTerms.end());
	}

	// Jacobian sparsity/terms
	{
		// (row,col) -> (chem_idx, kind) -> duplicacy
		std::map<std::pair<int, int>, std::map<std::pair<int, int>, int>> jacMapping;
		auto addTerm = [&](int row, int col, int chem_idx, int kind, int duplicate_count) {
			std::pair<int, int> key1 = {row, col};
			std::pair<int, int> key2 = {chem_idx, kind};
			jacMapping[key1][key2] += duplicate_count;
		};

		for (const auto& mp : dataDict) {
			int init = std::stoi(mp.at("init_assem_id"));
			int entering = is_stoiable(mp.at("entering_assem_id")) ? std::stoi(mp.at("entering_assem_id")) : config::species;
			int product = std::stoi(mp.at("product_assem_id"));
			int leaving = is_stoiable(mp.at("leaving_assem_id")) ? std::stoi(mp.at("leaving_assem_id")) : config::species;
			int kind = termIndex.at(mp.at("kind"));
			int duplicacy = std::stoi(mp.at("duplicate_count"));

			addTerm(init, init, entering, kind, -duplicacy);
			if (entering != config::species) {
				addTerm(init, entering, init, kind, -duplicacy);
			}
			if (entering != config::species) {
				addTerm(entering, init, entering, kind, -duplicacy);
				addTerm(entering, entering, init, kind, -duplicacy);
			}
			addTerm(product, init, entering, kind, duplicacy);
			if (entering != config::species) {
				addTerm(product, entering, init, kind, duplicacy);
			}
			if (leaving != config::species) {
				addTerm(leaving, init, entering, kind, duplicacy);
				if (entering != config::species) {
					addTerm(leaving, entering, init, kind, duplicacy);
				}
			}
		}

		int cnt = 0;
		jacNonZeros = 0;
		jacIdxPointer.resize(config::species + 1);
		jacIdxValue.clear();
		jacTerms.clear();

		for (int col = 0; col < config::species; ++col) {
			jacIdxPointer[col] = cnt;
			for (const auto& [key1, mp] : jacMapping) {
				int row = key1.first;
				int key_col = key1.second;
				if (key_col != col) continue;

				jacIdxValue.push_back(row);
				std::vector<JacTerm> termList;
				termList.reserve(mp.size());
				for (const auto& [key2, dup] : mp) {
					JacTerm t;
					t.reactant = key2.first;
					t.rateConstant = key2.second;
					t.duplicacy = dup;
					assert(0 <= t.reactant && t.reactant <= config::species);
					termList.push_back(t);
				}
				std::sort(termList.begin(), termList.end());
				jacTerms.push_back(std::move(termList));
				++cnt;
				++jacNonZeros;
			}
		}
		jacIdxPointer[config::species] = cnt;
	}
}

int ReactionNetwork::rhsfCb(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
	auto* ud = static_cast<CvodeUserData*>(user_data);
	if (ud == nullptr || ud->net == nullptr || ud->constants == nullptr) return -1;
	return ud->net->rhsfImpl(t, y, ydot, ud->constants);
}

int ReactionNetwork::JacFnCb(sunrealtype t,
						N_Vector y,
						N_Vector fy,
						SUNMatrix Jac,
						void* user_data,
						N_Vector tmp1,
						N_Vector tmp2,
						N_Vector tmp3) {
	auto* ud = static_cast<CvodeUserData*>(user_data);
	if (ud == nullptr || ud->net == nullptr || ud->constants == nullptr) return -1;
	return ud->net->jacImpl(t, y, fy, Jac, ud->constants);
}

int ReactionNetwork::rhsfImpl(sunrealtype /*t*/, N_Vector y, N_Vector ydot, const double* constants) {
	auto* sp_ptr = N_VGetArrayPointer(y);
	auto* ydotData = N_VGetArrayPointer(ydot);
	assert(sp_ptr != nullptr);
	assert(ydotData != nullptr);
	assert(constants != nullptr);

	std::memcpy(speciesData_.data(), sp_ptr, config::species * sizeof(double));
	speciesData_[config::species] = 1.0;
	std::fill(ydotData, ydotData + config::species, 0.0);

	std::span<double> ydotspan(ydotData, config::species);
	std::span<const double> speciesSpan(speciesData_.data(), config::species + 1);
	std::span<const double> kSpan(constants, config::constantSize);

	for (const auto& term : rhsTerms) {
		term.accumulate(ydotspan, speciesSpan, kSpan);
	}
	return 0;
}

int ReactionNetwork::jacImpl(sunrealtype /*t*/,
					N_Vector y,
					N_Vector /*fy*/,
					SUNMatrix Jac,
					const double* constants) {
	auto* sp_ptr = N_VGetArrayPointer(y);
	assert(sp_ptr != nullptr);
	assert(constants != nullptr);

	sunindextype* Jp = SUNSparseMatrix_IndexPointers(Jac);
	sunindextype* Ji = SUNSparseMatrix_IndexValues(Jac);
	sunrealtype* Jx = SUNSparseMatrix_Data(Jac);
	assert(Jp != nullptr && Ji != nullptr && Jx != nullptr);

	for (int i = 0; i <= config::species; ++i) {
		Jp[i] = jacIdxPointer[i];
	}
	for (int i = 0; i < jacNonZeros; ++i) {
		Ji[i] = jacIdxValue[i];
	}

	std::memcpy(speciesData_.data(), sp_ptr, config::species * sizeof(double));
	speciesData_[config::species] = 1.0;

	std::span<const double> speciesSpan(speciesData_.data(), config::species + 1);
	std::span<const double> kSpan(constants, config::constantSize);

	int idx = 0;
	for (const auto& termList : jacTerms) {
		double val = 0.0;
		for (const auto& t : termList) {
			val += t.value(speciesSpan, kSpan);
		}
		Jx[idx] = val;
		++idx;
	}
	return 0;
}
