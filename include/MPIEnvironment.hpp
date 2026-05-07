
#pragma once

// This header is designed to be usable even when MPI headers are unavailable.
// If <mpi.h> is not present, we provide minimal stubs so code can compile and
// run in single-process mode.

#if defined(__has_include)
#  if __has_include(<mpi.h>)
#    define NASAP_FIT_HAS_MPI 1
#    include <mpi.h>
#  else
#    define NASAP_FIT_HAS_MPI 0
#  endif
#else
#  define NASAP_FIT_HAS_MPI 0
#endif

#if !NASAP_FIT_HAS_MPI
using MPI_Comm = int;
inline constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline constexpr int MPI_THREAD_FUNNELED = 1;

inline int MPI_Initialized(int* flag) {
  if (flag) *flag = 0;
  return 0;
}

inline int MPI_Finalized(int* flag) {
  if (flag) *flag = 0;
  return 0;
}

inline int MPI_Comm_rank(MPI_Comm, int* rank) {
  if (rank) *rank = 0;
  return 0;
}

inline int MPI_Comm_size(MPI_Comm, int* size) {
  if (size) *size = 1;
  return 0;
}

inline int MPI_Query_thread(int* provided) {
  if (provided) *provided = 0;
  return 0;
}

inline int MPI_Init_thread(int*, char***, int required, int* provided) {
  if (provided) *provided = required;
  return 0;
}

inline int MPI_Finalize() { return 0; }
#endif

class MpiEnvironment {
public:
  // Does NOT call MPI_Init*. If MPI is already initialized, queries rank/size.
  explicit MpiEnvironment(MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
      MPI_Comm_rank(comm_, &rank_);
      MPI_Comm_size(comm_, &size_);
      MPI_Query_thread(&thread_support_);
    }
  }

  // Calls MPI_Init_thread only if MPI isn't initialized yet.
  MpiEnvironment(int& argc, char**& argv, int required = MPI_THREAD_FUNNELED, MPI_Comm comm = MPI_COMM_WORLD)
      : comm_(comm) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
      int provided = 0;
      MPI_Init_thread(&argc, &argv, required, &provided);
      owns_ = true;
      thread_support_ = provided;
    } else {
      MPI_Query_thread(&thread_support_);
    }
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }

  ~MpiEnvironment() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (owns_ && !finalized) MPI_Finalize();
  }

  MPI_Comm comm() const { return comm_; }
  int rank() const { return rank_; }
  int size() const { return size_; }
  int thread_support() const { return thread_support_; }

private:
  bool owns_{false};
  MPI_Comm comm_{MPI_COMM_WORLD};
  int rank_{0};
  int size_{1};
  int thread_support_{0};
};