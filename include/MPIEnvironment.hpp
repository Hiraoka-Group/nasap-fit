
#pragma once

#include <mpi.h>

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