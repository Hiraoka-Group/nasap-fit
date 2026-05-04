#pragma once

#include <algorithm>
#include <charconv>
#include <compare>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>

struct ReactionNetwork {
	int species = 0;
	int constantSize = 0;

	std::vector<std::array<int, 6>>data;

	struct RhsTerm {
		int add_to{};
		int duplicacy{};
		int reactant1{};
		int reactant2{}; // dummy index means "no second reactant"
		int rateConstant{};

		inline void accumulate(std::span<double> ydot,
							std::span<const double> species,
							std::span<const double> k) const {
			ydot[add_to] += duplicacy * k[rateConstant] * species[reactant1] * species[reactant2];
		}

		auto operator<=>(const RhsTerm&) const;
	};

	struct JacTerm {
		int duplicacy{};
		int reactant{}; // dummy index means "no reactant" (dummy=1.0)
		int rateConstant{};

		inline double value(std::span<const double> species, std::span<const double> k) const {
			return duplicacy * k[rateConstant] * species[reactant];
		}

		auto operator<=>(const JacTerm&) const;
	};

	struct CvodeUserData {
		ReactionNetwork* net{};
		const double* p{}; // length: constantSize
		const std::vector<int>* reactionIds;
		int Ns = 0;
		const int* plist = nullptr; // length: Ns, maps sensitivity index -> parameter index
		
	};

	std::map<std::string, int> termIndex;      // kind -> index
	std::vector<RhsTerm> rhsTerms;            // mass-action style RHS terms

	int jacNonZeros = 0;                      // number of non-zeros in sparse Jacobian
	std::vector<std::vector<JacTerm>> jacTerms; // per-nonzero entry terms
	std::vector<int> jacIdxPointer;           // CSC column pointer (size species+1)
	std::vector<int> jacIdxValue;             // CSC row indices (size jacNonZeros)

	ReactionNetwork() = default;

	// Build RHS/Jacobian structures from a reaction-network CSV file.
	// species/constantSize are injected so this type doesn't depend on global config.
	void build(const std::string& reactNetworkFile, int species, int constantSize);

	// CVODE callback wrappers (used when USE_PREGENERATED_* == 0)
	static int rhsfCb(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
	static int JacFnCb(sunrealtype t,
					N_Vector y,
					N_Vector fy,
					SUNMatrix Jac,
					void* user_data,
					N_Vector tmp1,
					N_Vector tmp2,
					N_Vector tmp3);

	static int quadRhsCb(sunrealtype t, N_Vector y, N_Vector yQdot, void *user_data);
	static int sensRhsCb(int Ns,
					sunrealtype t,
					N_Vector y,
					N_Vector ydot,
					N_Vector* yS,
					N_Vector* ySdot,
					void* user_data,
					N_Vector tmp1,
					N_Vector tmp2);

private:
	std::vector<double> speciesData_; // size: species + 1 (last is dummy=1.0)

	void ensureSizes();

	int rhsfImpl(sunrealtype t, N_Vector y, N_Vector ydot, const double* p);
	int jacImpl(sunrealtype t,
			N_Vector y,
			N_Vector fy,
			SUNMatrix Jac,
			const double* p);
	
	int quadRhsImpl(sunrealtype t, N_Vector y, N_Vector yQdot, const double* p, const std::vector<int>* reactionIds);
	int sensRhsImpl(int Ns,
				sunrealtype t,
				N_Vector y,
				N_Vector ydot,
				N_Vector* yS,
				N_Vector* ySdot,
				const double* p,
				const int* plist);
};
