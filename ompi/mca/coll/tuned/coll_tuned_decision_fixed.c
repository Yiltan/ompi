/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "cuda_runtime.h"
#include "opal/util/bit_ops.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/op/op.h"
#include "coll_tuned.h"

#define NUM_GPUS 4

int comms_initialised = 0;
int use_hierarchical_allreduce = 0;
int intra_allreduce_algo = 0;
int intra_reduce_algo = -1;
int intra_bcast_algo = -1;
int inter_algo = 0;

struct ompi_communicator_t* intra_comm;
struct ompi_communicator_t* inter_comm;

struct ompi_communicator_t* comm_cache;

// This is a hash function from: http://www.cse.yorku.ca/~oz/hash.html
static inline int hash(char *input, int len) {
    unsigned char *str = (unsigned char *) input;
    unsigned long hash = 5381;
    int c;

    while (c = *str++) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
        len = len - 1;
    }

    return abs((int) hash);
}

static inline void init_comms(struct ompi_communicator_t* original_comm) {
    if (comm_cache != original_comm) {
        comm_cache = original_comm;

        int key = original_comm->c_my_rank;
        int rank = ompi_comm_rank(original_comm);
        int color;

        // Get Intra comm
        char name[MPI_MAX_PROCESSOR_NAME];
        int resultlen;
        MPI_Get_processor_name(name, &resultlen);

        ompi_comm_split(original_comm, hash(name, resultlen), key,
                        &intra_comm, false);

        if (0 == rank) {
          printf("Intra_comm size = %d\n", ompi_comm_size(intra_comm));
        }

        // Get Inter group
        ompi_comm_split(original_comm, ompi_comm_rank(intra_comm), key,
                        &inter_comm, false);

        if (0 == rank) {
          printf("inter_comm size = %d\n", ompi_comm_size(inter_comm));
        }

        char *env = getenv("USE_HIERARCHICAL_ALLREDUCE");
        if (NULL != env) {
          use_hierarchical_allreduce = atoi(env);
        }

        env = getenv("ALLREDUCE_INTRA_ALGO");
        if (NULL != env) {
          intra_allreduce_algo = atoi(env);
        }

        env = getenv("ALLREDUCE_INTER_ALGO");
        if (NULL != env) {
          inter_algo = atoi(env);
        }

        env = getenv("BCAST_INTRA_ALGO");
        if (NULL != env) {
          intra_bcast_algo = atoi(env);
        }

        env = getenv("REDUCE_INTRA_ALGO");
        if (NULL != env) {
          intra_reduce_algo = atoi(env);
        }
    }
}

#define ALLREDUCE_INTRA_NONOVERLAPPING    0
#define ALLREDUCE_INTRA_RECURSIVEDOUBLING 1
#define ALLREDUCE_INTRA_RING              2
#define ALLREDUCE_INTRA_RING_SEGMENTED    3
#define ALLREDUCE_INTRA_BASIC_LINEAR      4
#define ALLREDUCE_INTRA_REDSCAT_ALLGATHER 5
#define ALLREDUCE_INTRA_YHT               6

int allreduce_switch(const void *sbuf, void *rbuf, int count,
                     struct ompi_datatype_t *dtype,
                     struct ompi_op_t *op,
                     struct ompi_communicator_t *comm,
                     mca_coll_base_module_t *module,
                     uint32_t segsize,
                     int coll_num) {
  switch (coll_num) {
    case ALLREDUCE_INTRA_NONOVERLAPPING:
      return ompi_coll_base_allreduce_intra_nonoverlapping(sbuf, rbuf, count,
                                                           dtype, op, comm, module);
      break;
    case ALLREDUCE_INTRA_RECURSIVEDOUBLING:
      return ompi_coll_base_allreduce_intra_recursivedoubling(sbuf, rbuf, count,
                                                              dtype, op, comm, module);
      break;
    case ALLREDUCE_INTRA_RING:
      return ompi_coll_base_allreduce_intra_ring(sbuf, rbuf, count, dtype, op, comm, module);
      break;
    case ALLREDUCE_INTRA_RING_SEGMENTED:
      return ompi_coll_base_allreduce_intra_ring_segmented(sbuf, rbuf, count,
                                                           dtype, op, comm, module, segsize);
      break;
    case ALLREDUCE_INTRA_BASIC_LINEAR:
      return ompi_coll_base_allreduce_intra_basic_linear(sbuf, rbuf, count,
                                                         dtype, op, comm, module);
      break;
    case ALLREDUCE_INTRA_REDSCAT_ALLGATHER:
      return ompi_coll_base_allreduce_intra_redscat_allgather(sbuf, rbuf, count,
                                                              dtype, op, comm, module);
      break;
    case ALLREDUCE_INTRA_YHT:
    default:
      return ompi_coll_base_allreduce_intra_yht(sbuf, rbuf, count, dtype, op, comm, module);
      break;
  }
}

#define BCAST_INTRA_BINOMIAL      0
#define BCAST_INTRA_PIPELINE      1
#define BCAST_INTRA_SPLIT_BINTREE 2

int bcast_switch(void *buff, int count,
                 struct ompi_datatype_t *datatype, int root,
                 struct ompi_communicator_t *comm,
                 mca_coll_base_module_t *module,
                 int coll_num,
                 int segsize) {

  switch (coll_num) {
    case BCAST_INTRA_BINOMIAL:
        return ompi_coll_base_bcast_intra_binomial(buff, count, datatype, root, comm,
                                                   module, segsize);
    case BCAST_INTRA_PIPELINE:
        return ompi_coll_base_bcast_intra_pipeline(buff, count, datatype, root, comm,
                                                   module, segsize);
    case BCAST_INTRA_SPLIT_BINTREE:
    default:
        return ompi_coll_base_bcast_intra_split_bintree(buff, count, datatype, root, comm,
                                                        module, segsize);
  }
}

#define REDUCE_INTRA_BASIC_LINEAR    0
#define REDUCE_INTRA_BINARY          1
#define REDUCE_INTRA_BINOMIAL        2
#define REDUCE_INTRA_IN_ORDER_BINARY 3
#define REDUCE_INTRA_PIPELINE        4

int debug_switch = 0;

int reduce_switch(const void *sendbuf, void *recvbuf,
                  int count, struct ompi_datatype_t* datatype,
                  struct ompi_op_t* op, int root,
                  struct ompi_communicator_t* comm,
                  mca_coll_base_module_t *module,
                  int coll_num,
                  int segsize) {

  const int max_requests = 0; /* no limit on # of outstanding requests */

  switch (coll_num) {
    case REDUCE_INTRA_BASIC_LINEAR:
      return ompi_coll_base_reduce_intra_basic_linear(sendbuf, recvbuf, count, datatype, op,
                                                      root, comm, module);
    case REDUCE_INTRA_BINARY:
      return ompi_coll_base_reduce_intra_binary(sendbuf, recvbuf, count, datatype, op,
                                                root, comm, module, segsize, max_requests);
    case REDUCE_INTRA_BINOMIAL:
      return ompi_coll_base_reduce_intra_binomial(sendbuf, recvbuf, count, datatype, op,
                                                  root, comm, module, segsize, max_requests);
    case REDUCE_INTRA_IN_ORDER_BINARY:
      return ompi_coll_base_reduce_intra_in_order_binary(sendbuf, recvbuf, count, datatype, op,
                                                         root, comm, module, 0, max_requests);
    case REDUCE_INTRA_PIPELINE:
    default:
      return ompi_coll_base_reduce_intra_pipeline(sendbuf, recvbuf, count, datatype, op,
                                                  root, comm, module, segsize, max_requests);
  }
}

/*
 *  allreduce_intra
 *
 *  Function:   - allreduce using other MPI collectives
 *  Accepts:    - same as MPI_Allreduce()
 *  Returns:    - MPI_SUCCESS or error code
 */
int
ompi_coll_tuned_allreduce_intra_dec_fixed(const void *sbuf, void *rbuf, int count,
                                          struct ompi_datatype_t *dtype,
                                          struct ompi_op_t *op,
                                          struct ompi_communicator_t *comm,
                                          mca_coll_base_module_t *module)
{
    size_t dsize, block_dsize;
    int comm_size = ompi_comm_size(comm);
    const size_t intermediate_message = 10000;
    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_allreduce_intra_dec_fixed"));

    init_comms(comm);

    ompi_datatype_type_size(dtype, &dsize);
    block_dsize = dsize * (ptrdiff_t)count;

    // If single node (assuming 4 PPN) or using a flat algo
    // then use original algo.
    // This can be optimized later for single node if needed.
    const size_t segment_size = 1 << 20; /* 1 MB */
    if (0 == use_hierarchical_allreduce) {
        if (block_dsize < intermediate_message) {
            return (ompi_coll_base_allreduce_intra_recursivedoubling(sbuf, rbuf,
                                                                     count, dtype,
                                                                     op, comm,
                                                                     module));
        }

        if( ompi_op_is_commute(op) && (count > comm_size) ) {
            if (((size_t)comm_size * (size_t)segment_size >= block_dsize)) {
                return (ompi_coll_base_allreduce_intra_ring(sbuf, rbuf, count,
                                                            dtype, op, comm,
                                                            module));
            } else {
                return (ompi_coll_base_allreduce_intra_ring_segmented(sbuf, rbuf,
                                                                      count, dtype,
                                                                      op, comm,
                                                                      module,
                                                                      segment_size));
            }
        }

        return (ompi_coll_base_allreduce_intra_nonoverlapping(sbuf, rbuf, count,
                                                              dtype, op, comm,
                                                              module));
    } else if (1 == use_hierarchical_allreduce) {
        // 2 stage algo
        int ret_val = MPI_SUCCESS;
        ret_val &= (allreduce_switch(sbuf, rbuf, count, dtype, op, inter_comm,
                                    module, segment_size, inter_algo));

        ret_val &= (allreduce_switch(MPI_IN_PLACE, rbuf, count, dtype, op,
                                     intra_comm, module, segment_size, intra_allreduce_algo));
        return ret_val;
    } else if (2 == use_hierarchical_allreduce) {
        // 3 stage algo
        int segsize = 1024;
        int ret_val = MPI_SUCCESS;

        if (MPI_IN_PLACE == sbuf) {
            ret_val &= (ompi_coll_tuned_reduce_intra_dec_fixed(rbuf, rbuf, count,
                                                               dtype, op, 0,
                                                               intra_comm, module));
        }
        else {
            ret_val &= (ompi_coll_tuned_reduce_intra_dec_fixed(sbuf, rbuf, count,
                                                               dtype, op, 0,
                                                               intra_comm, module));
        }

        if (0 == ompi_comm_rank(intra_comm)) {
          ret_val &= (allreduce_switch(MPI_IN_PLACE, rbuf, count, dtype, op,
                                       inter_comm, module, segment_size,
                                       inter_algo));
        }

        ret_val &= (ompi_coll_tuned_bcast_intra_dec_fixed(rbuf, count, dtype, 0,
                                                          intra_comm, module));
        return ret_val;
    } else if (3 == use_hierarchical_allreduce) {
        // Flat algo
        return (allreduce_switch(sbuf, rbuf, count, dtype, op,
                                 comm, module, segment_size,
                                 intra_allreduce_algo));
    } else if (4 == use_hierarchical_allreduce) {
        // Flat algo
        return (allreduce_switch(sbuf, rbuf, count, dtype, op,
                                 intra_comm, module, segment_size,
                                 intra_allreduce_algo));
    } else {
        return -1;
    }
}

/*
 *	alltoall_intra_dec
 *
 *	Function:	- seletects alltoall algorithm to use
 *	Accepts:	- same arguments as MPI_Alltoall()
 *	Returns:	- MPI_SUCCESS or error code
 */

int ompi_coll_tuned_alltoall_intra_dec_fixed(const void *sbuf, int scount,
                                             struct ompi_datatype_t *sdtype,
                                             void* rbuf, int rcount,
                                             struct ompi_datatype_t *rdtype,
                                             struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    int communicator_size;
    size_t dsize, block_dsize;
#if 0
    size_t total_dsize;
#endif

    communicator_size = ompi_comm_size(comm);

    /* special case */
    if (communicator_size==2) {
        return ompi_coll_base_alltoall_intra_two_procs(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       comm, module);
    }

    /* Decision function based on measurement on Grig cluster at
       the University of Tennessee (2GB MX) up to 64 nodes.
       Has better performance for messages of intermediate sizes than the old one */
    /* determine block size */
    if (MPI_IN_PLACE != sbuf) {
        ompi_datatype_type_size(sdtype, &dsize);
    } else {
        ompi_datatype_type_size(rdtype, &dsize);
    }
    block_dsize = dsize * (ptrdiff_t)scount;

    if ((block_dsize < (size_t) ompi_coll_tuned_alltoall_small_msg)
                                              && (communicator_size > 12)) {
        return ompi_coll_base_alltoall_intra_bruck(sbuf, scount, sdtype,
                                                   rbuf, rcount, rdtype,
                                                   comm, module);

    } else if (block_dsize < (size_t) ompi_coll_tuned_alltoall_intermediate_msg) {
        return ompi_coll_base_alltoall_intra_basic_linear(sbuf, scount, sdtype,
                                                          rbuf, rcount, rdtype,
                                                          comm, module);
    } else if ((block_dsize < (size_t) ompi_coll_tuned_alltoall_large_msg) &&
               (communicator_size <= ompi_coll_tuned_alltoall_min_procs)) {
        return ompi_coll_base_alltoall_intra_linear_sync(sbuf, scount, sdtype,
                                                         rbuf, rcount, rdtype,
                                                         comm, module,
                                                         ompi_coll_tuned_alltoall_max_requests);
    }

    return ompi_coll_base_alltoall_intra_pairwise(sbuf, scount, sdtype,
                                                  rbuf, rcount, rdtype,
                                                  comm, module);

#if 0
    /* previous decision */

    /* else we need data size for decision function */
    ompi_datatype_type_size(sdtype, &dsize);
    total_dsize = dsize * scount * communicator_size;   /* needed for decision */

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_alltoall_intra_dec_fixed rank %d com_size %d msg_length %ld",
                 ompi_comm_rank(comm), communicator_size, total_dsize));

    if (communicator_size >= 12 && total_dsize <= 768) {
        return ompi_coll_base_alltoall_intra_bruck(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, module);
    }
    if (total_dsize <= 131072) {
        return ompi_coll_base_alltoall_intra_basic_linear(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, module);
    }
    return ompi_coll_base_alltoall_intra_pairwise(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, module);
#endif
}

/*
 *      Function:       - selects alltoallv algorithm to use
 *      Accepts:        - same arguments as MPI_Alltoallv()
 *      Returns:        - MPI_SUCCESS or error code
 */
int ompi_coll_tuned_alltoallv_intra_dec_fixed(const void *sbuf, const int *scounts, const int *sdisps,
                                              struct ompi_datatype_t *sdtype,
                                              void *rbuf, const int *rcounts, const int *rdisps,
                                              struct ompi_datatype_t *rdtype,
                                              struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module)
{
    /* For starters, just keep the original algorithm. */
    return ompi_coll_base_alltoallv_intra_pairwise(sbuf, scounts, sdisps, sdtype,
                                                   rbuf, rcounts, rdisps,rdtype,
                                                   comm, module);
}


/*
 *	barrier_intra_dec
 *
 *	Function:	- seletects barrier algorithm to use
 *	Accepts:	- same arguments as MPI_Barrier()
 *	Returns:	- MPI_SUCCESS or error code (passed from the barrier implementation)
 */
int ompi_coll_tuned_barrier_intra_dec_fixed(struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    int communicator_size = ompi_comm_size(comm);

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_barrier_intra_dec_fixed com_size %d",
                 communicator_size));

    if( 2 == communicator_size )
        return ompi_coll_base_barrier_intra_two_procs(comm, module);
    /**
     * Basic optimisation. If we have a power of 2 number of nodes
     * the use the recursive doubling algorithm, otherwise
     * bruck is the one we want.
     */
    {
        bool has_one = false;
        for( ; communicator_size > 0; communicator_size >>= 1 ) {
            if( communicator_size & 0x1 ) {
                if( has_one )
                    return ompi_coll_base_barrier_intra_bruck(comm, module);
                has_one = true;
            }
        }
    }
    return ompi_coll_base_barrier_intra_recursivedoubling(comm, module);
}


/*
 *	bcast_intra_dec
 *
 *	Function:	- seletects broadcast algorithm to use
 *	Accepts:	- same arguments as MPI_Bcast()
 *	Returns:	- MPI_SUCCESS or error code (passed from the bcast implementation)
 */
int ompi_coll_tuned_bcast_intra_dec_fixed(void *buff, int count,
                                          struct ompi_datatype_t *datatype, int root,
                                          struct ompi_communicator_t *comm,
                                          mca_coll_base_module_t *module)
{
    /* Decision function based on MX results for
       messages up to 36MB and communicator sizes up to 64 nodes */
    const size_t small_message_size = 2048;
    const size_t intermediate_message_size = 370728;
    const double a_p16  = 3.2118e-6; /* [1 / byte] */
    const double b_p16  = 8.7936;
    const double a_p64  = 2.3679e-6; /* [1 / byte] */
    const double b_p64  = 1.1787;
    const double a_p128 = 1.6134e-6; /* [1 / byte] */
    const double b_p128 = 2.1102;

    int communicator_size;
    int segsize = 0;
    size_t message_size, dsize;

    communicator_size = ompi_comm_size(comm);
    init_comms(comm);

    /* else we need data size for decision function */
    ompi_datatype_type_size(datatype, &dsize);
    message_size = dsize * (unsigned long)count;   /* needed for decision */

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_bcast_intra_dec_fixed"
                 " root %d rank %d com_size %d msg_length %lu",
                 root, ompi_comm_rank(comm), communicator_size, (unsigned long)message_size));
    if (-1 == intra_bcast_algo) {

      /* Handle messages of small and intermediate size, and
         single-element broadcasts */
      if ((message_size < small_message_size) || (count <= 1)) {
          /* Binomial without segmentation */
          segsize = 0;
          return  ompi_coll_base_bcast_intra_binomial(buff, count, datatype,
                                                      root, comm, module,
                                                      segsize);

      } else if (message_size < intermediate_message_size) {
          /* SplittedBinary with 1KB segments */
          segsize = 1024;
          return ompi_coll_base_bcast_intra_split_bintree(buff, count, datatype,
                                                          root, comm, module,
                                                          segsize);

      }
      /* Handle large message sizes */
      else if (communicator_size < (a_p128 * message_size + b_p128)) {
          /* Pipeline with 128KB segments */
          segsize = 1024  << 7;
          return ompi_coll_base_bcast_intra_pipeline(buff, count, datatype,
                                                     root, comm, module,
                                                     segsize);

      } else if (communicator_size < 13) {
          /* Split Binary with 8KB segments */
          segsize = 1024 << 3;
          return ompi_coll_base_bcast_intra_split_bintree(buff, count, datatype,
                                                          root, comm, module,
                                                          segsize);

      } else if (communicator_size < (a_p64 * message_size + b_p64)) {
          /* Pipeline with 64KB segments */
          segsize = 1024 << 6;
          return ompi_coll_base_bcast_intra_pipeline(buff, count, datatype,
                                                     root, comm, module,
                                                     segsize);

      } else if (communicator_size < (a_p16 * message_size + b_p16)) {
          /* Pipeline with 16KB segments */
          segsize = 1024 << 4;
          return ompi_coll_base_bcast_intra_pipeline(buff, count, datatype,
                                                     root, comm, module,
                                                     segsize);

      }

      /* Pipeline with 8KB segments */
      segsize = 1024 << 3;
      return ompi_coll_base_bcast_intra_pipeline(buff, count, datatype,
                                                 root, comm, module,
                                                 segsize);
    } else {
      if ((message_size < small_message_size) || (count <= 1)) {
          segsize = 0;
      } else if (message_size < intermediate_message_size) {
          /* SplittedBinary with 1KB segments */
          segsize = 1024;
      }
      /* Handle large message sizes */
      else if (communicator_size < (a_p128 * message_size + b_p128)) {
          /* Pipeline with 128KB segments */
          segsize = 1024  << 7;
      } else if (communicator_size < 13) {
          /* Split Binary with 8KB segments */
          segsize = 1024 << 3;
      } else if (communicator_size < (a_p64 * message_size + b_p64)) {
          /* Pipeline with 64KB segments */
          segsize = 1024 << 6;
      } else if (communicator_size < (a_p16 * message_size + b_p16)) {
          /* Pipeline with 16KB segments */
          segsize = 1024 << 4;
      } else {
        /* Pipeline with 8KB segments */
        segsize = 1024 << 3;
      }

      return bcast_switch(buff, count, datatype, root, comm,
                          module, intra_bcast_algo, segsize);
    }
#if 0
    /* this is based on gige measurements */

    if (communicator_size  < 4) {
        return ompi_coll_base_bcast_intra_basic_linear(buff, count, datatype, root, comm, module);
    }
    if (communicator_size == 4) {
        if (message_size < 524288) segsize = 0;
        else segsize = 16384;
        return ompi_coll_base_bcast_intra_bintree(buff, count, datatype, root, comm, module, segsize);
    }
    if (communicator_size <= 8 && message_size < 4096) {
        return ompi_coll_base_bcast_intra_basic_linear(buff, count, datatype, root, comm, module);
    }
    if (communicator_size > 8 && message_size >= 32768 && message_size < 524288) {
        segsize = 16384;
        return  ompi_coll_base_bcast_intra_bintree(buff, count, datatype, root, comm, module, segsize);
    }
    if (message_size >= 524288) {
        segsize = 16384;
        return ompi_coll_base_bcast_intra_pipeline(buff, count, datatype, root, comm, module, segsize);
    }
    segsize = 0;
    /* once tested can swap this back in */
    /* return ompi_coll_base_bcast_intra_bmtree(buff, count, datatype, root, comm, segsize); */
    return ompi_coll_base_bcast_intra_bintree(buff, count, datatype, root, comm, module, segsize);
#endif  /* 0 */
}

/*
 *	reduce_intra_dec
 *
 *	Function:	- seletects reduce algorithm to use
 *	Accepts:	- same arguments as MPI_reduce()
 *	Returns:	- MPI_SUCCESS or error code (passed from the reduce implementation)
 *
 */
int ompi_coll_tuned_reduce_intra_dec_fixed( const void *sendbuf, void *recvbuf,
                                            int count, struct ompi_datatype_t* datatype,
                                            struct ompi_op_t* op, int root,
                                            struct ompi_communicator_t* comm,
                                            mca_coll_base_module_t *module)
{
    int communicator_size, segsize = 0;
    size_t message_size, dsize;
    const double a1 =  0.6016 / 1024.0; /* [1/B] */
    const double b1 =  1.3496;
    const double a2 =  0.0410 / 1024.0; /* [1/B] */
    const double b2 =  9.7128;
    const double a3 =  0.0422 / 1024.0; /* [1/B] */
    const double b3 =  1.1614;
    const double a4 =  0.0033 / 1024.0; /* [1/B] */
    const double b4 =  1.6761;

    const int max_requests = 0; /* no limit on # of outstanding requests */

    init_comms(comm);

    communicator_size = ompi_comm_size(comm);

    /* need data size for decision function */
    ompi_datatype_type_size(datatype, &dsize);
    message_size = dsize * (ptrdiff_t)count;   /* needed for decision */

    if (-1 == intra_reduce_algo) {
      /**
       * If the operation is non commutative we currently have choice of linear
       * or in-order binary tree algorithm.
       */
      if( !ompi_op_is_commute(op) ) {
          if ((communicator_size < 12) && (message_size < 2048)) {
              return ompi_coll_base_reduce_intra_basic_linear (sendbuf, recvbuf, count, datatype, op, root, comm, module);
          }
          return ompi_coll_base_reduce_intra_in_order_binary (sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                               0, max_requests);
      }

      OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_reduce_intra_dec_fixed "
                   "root %d rank %d com_size %d msg_length %lu",
                   root, ompi_comm_rank(comm), communicator_size, (unsigned long)message_size));

      if ((communicator_size < 8) && (message_size < 512)){
          /* Linear_0K */
          return ompi_coll_base_reduce_intra_basic_linear(sendbuf, recvbuf, count, datatype, op, root, comm, module);
      } else if (((communicator_size < 8) && (message_size < 20480)) ||
                 (message_size < 2048) || (count <= 1)) {
          /* Binomial_0K */
          segsize = 0;
          return ompi_coll_base_reduce_intra_binomial(sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                       segsize, max_requests);
      } else if (communicator_size > (a1 * message_size + b1)) {
          /* Binomial_1K */
          segsize = 1024;
          return ompi_coll_base_reduce_intra_binomial(sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                       segsize, max_requests);
      } else if (communicator_size > (a2 * message_size + b2)) {
          /* Pipeline_1K */
          segsize = 1024;
          return ompi_coll_base_reduce_intra_pipeline(sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                      segsize, max_requests);
      } else if (communicator_size > (a3 * message_size + b3)) {
          /* Binary_32K */
          segsize = 32*1024;
          return ompi_coll_base_reduce_intra_binary( sendbuf, recvbuf, count, datatype, op, root,
                                                      comm, module, segsize, max_requests);
      }
      if (communicator_size > (a4 * message_size + b4)) {
          /* Pipeline_32K */
          segsize = 32*1024;
      } else {
          /* Pipeline_64K */
          segsize = 64*1024;
      }
      return ompi_coll_base_reduce_intra_pipeline(sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                  segsize, max_requests);
    } else {
      if (communicator_size > (a1 * message_size + b1) ||
          communicator_size > (a2 * message_size + b2)) {
          segsize = 1024;
      } else if (communicator_size > (a3 * message_size + b3) ||
                 communicator_size > (a4 * message_size + b4)) {
          segsize = 32*1024;
      } else {
          segsize = 64*1024;
      }
      return reduce_switch(sendbuf, recvbuf, count, datatype, op, root,
                           comm, module, intra_reduce_algo, segsize);
    }

#if 0
    /* for small messages use linear algorithm */
    if (message_size <= 4096) {
        segsize = 0;
        fanout = communicator_size - 1;
        /* when linear implemented or taken from basic put here, right now using chain as a linear system */
        /* it is implemented and I shouldn't be calling a chain with a fanout bigger than MAXTREEFANOUT from topo.h! */
        return ompi_coll_base_reduce_intra_basic_linear(sendbuf, recvbuf, count, datatype, op, root, comm, module);
    }
    if (message_size < 524288) {
        if (message_size <= 65536 ) {
            segsize = 32768;
            fanout = 8;
        } else {
            segsize = 1024;
            fanout = communicator_size/2;
        }
        /* later swap this for a binary tree */
        /*         fanout = 2; */
        return ompi_coll_base_reduce_intra_chain(sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                 segsize, fanout, max_requests);
    }
    segsize = 1024;
    return ompi_coll_base_reduce_intra_pipeline(sendbuf, recvbuf, count, datatype, op, root, comm, module,
                                                segsize, max_requests);
#endif  /* 0 */
}

/*
 *	reduce_scatter_intra_dec
 *
 *	Function:	- seletects reduce_scatter algorithm to use
 *	Accepts:	- same arguments as MPI_Reduce_scatter()
 *	Returns:	- MPI_SUCCESS or error code (passed from
 *                        the reduce scatter implementation)
 */
int ompi_coll_tuned_reduce_scatter_intra_dec_fixed( const void *sbuf, void *rbuf,
                                                    const int *rcounts,
                                                    struct ompi_datatype_t *dtype,
                                                    struct ompi_op_t *op,
                                                    struct ompi_communicator_t *comm,
                                                    mca_coll_base_module_t *module)
{
    int comm_size, i, pow2;
    size_t total_message_size, dsize;
    const double a = 0.0012;
    const double b = 8.0;
    const size_t small_message_size = 12 * 1024;
    const size_t large_message_size = 256 * 1024;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_reduce_scatter_intra_dec_fixed"));

    comm_size = ompi_comm_size(comm);
    /* We need data size for decision function */
    ompi_datatype_type_size(dtype, &dsize);
    total_message_size = 0;
    for (i = 0; i < comm_size; i++) {
        total_message_size += rcounts[i];
    }

    if( !ompi_op_is_commute(op) ) {
        return ompi_coll_base_reduce_scatter_intra_nonoverlapping(sbuf, rbuf, rcounts,
                                                                  dtype, op,
                                                                  comm, module);
    }

    total_message_size *= dsize;

    /* compute the nearest power of 2 */
    pow2 = opal_next_poweroftwo_inclusive (comm_size);

    if ((total_message_size <= small_message_size) ||
        ((total_message_size <= large_message_size) && (pow2 == comm_size)) ||
        (comm_size >= a * total_message_size + b)) {
        return
            ompi_coll_base_reduce_scatter_intra_basic_recursivehalving(sbuf, rbuf, rcounts,
                                                                       dtype, op,
                                                                       comm, module);
    }
    return ompi_coll_base_reduce_scatter_intra_ring(sbuf, rbuf, rcounts,
                                                     dtype, op,
                                                     comm, module);
}

/*
 *	reduce_scatter_block_intra_dec
 *
 *	Function:	- seletects reduce_scatter_block algorithm to use
 *	Accepts:	- same arguments as MPI_Reduce_scatter_block()
 *	Returns:	- MPI_SUCCESS or error code (passed from
 *                        the reduce scatter implementation)
 */
int ompi_coll_tuned_reduce_scatter_block_intra_dec_fixed(const void *sbuf, void *rbuf,
                                                         int rcount,
                                                         struct ompi_datatype_t *dtype,
                                                         struct ompi_op_t *op,
                                                         struct ompi_communicator_t *comm,
                                                         mca_coll_base_module_t *module)
{
    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_reduce_scatter_block_intra_dec_fixed"));
    return ompi_coll_base_reduce_scatter_block_basic_linear(sbuf, rbuf, rcount,
                                                            dtype, op, comm, module);
}

/*
 *	allgather_intra_dec
 *
 *	Function:	- seletects allgather algorithm to use
 *	Accepts:	- same arguments as MPI_Allgather()
 *	Returns:	- MPI_SUCCESS or error code, passed from corresponding
 *                        internal allgather function.
 */

int ompi_coll_tuned_allgather_intra_dec_fixed(const void *sbuf, int scount,
                                              struct ompi_datatype_t *sdtype,
                                              void* rbuf, int rcount,
                                              struct ompi_datatype_t *rdtype,
                                              struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module)
{
    int communicator_size, pow2_size;
    size_t dsize, total_dsize;

    communicator_size = ompi_comm_size(comm);

    /* Special case for 2 processes */
    if (communicator_size == 2) {
        return ompi_coll_base_allgather_intra_two_procs(sbuf, scount, sdtype,
                                                        rbuf, rcount, rdtype,
                                                        comm, module);
    }

    /* Determine complete data size */
    if (MPI_IN_PLACE != sbuf) {
        ompi_datatype_type_size(sdtype, &dsize);
    } else {
        ompi_datatype_type_size(rdtype, &dsize);
    }
    total_dsize = dsize * (ptrdiff_t)scount * (ptrdiff_t)communicator_size;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_allgather_intra_dec_fixed"
                 " rank %d com_size %d msg_length %lu",
                 ompi_comm_rank(comm), communicator_size, (unsigned long)total_dsize));

    pow2_size = opal_next_poweroftwo_inclusive (communicator_size);

    /* Decision based on MX 2Gb results from Grig cluster at
       The University of Tennesse, Knoxville
       - if total message size is less than 50KB use either bruck or
       recursive doubling for non-power of two and power of two nodes,
       respectively.
       - else use ring and neighbor exchange algorithms for odd and even
       number of nodes, respectively.
    */
    if (total_dsize < 50000) {
        if (pow2_size == communicator_size) {
            return ompi_coll_base_allgather_intra_recursivedoubling(sbuf, scount, sdtype,
                                                                    rbuf, rcount, rdtype,
                                                                    comm, module);
        } else {
            return ompi_coll_base_allgather_intra_bruck(sbuf, scount, sdtype,
                                                        rbuf, rcount, rdtype,
                                                        comm, module);
        }
    } else {
        if (communicator_size % 2) {
            return ompi_coll_base_allgather_intra_ring(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       comm, module);
        } else {
            return  ompi_coll_base_allgather_intra_neighborexchange(sbuf, scount, sdtype,
                                                                    rbuf, rcount, rdtype,
                                                                    comm, module);
        }
    }

#if defined(USE_MPICH2_DECISION)
    /* Decision as in MPICH-2
       presented in Thakur et.al. "Optimization of Collective Communication
       Operations in MPICH", International Journal of High Performance Computing
       Applications, Vol. 19, No. 1, 49-66 (2005)
       - for power-of-two processes and small and medium size messages
       (up to 512KB) use recursive doubling
       - for non-power-of-two processes and small messages (80KB) use bruck,
       - for everything else use ring.
    */
    if ((pow2_size == communicator_size) && (total_dsize < 524288)) {
        return ompi_coll_base_allgather_intra_recursivedoubling(sbuf, scount, sdtype,
                                                                rbuf, rcount, rdtype,
                                                                comm, module);
    } else if (total_dsize <= 81920) {
        return ompi_coll_base_allgather_intra_bruck(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    comm, module);
    }
    return ompi_coll_base_allgather_intra_ring(sbuf, scount, sdtype,
                                               rbuf, rcount, rdtype,
                                               comm, module);
#endif  /* defined(USE_MPICH2_DECISION) */
}

/*
 *	allgatherv_intra_dec
 *
 *	Function:	- seletects allgatherv algorithm to use
 *	Accepts:	- same arguments as MPI_Allgatherv()
 *	Returns:	- MPI_SUCCESS or error code, passed from corresponding
 *                        internal allgatherv function.
 */

int ompi_coll_tuned_allgatherv_intra_dec_fixed(const void *sbuf, int scount,
                                               struct ompi_datatype_t *sdtype,
                                               void* rbuf, const int *rcounts,
                                               const int *rdispls,
                                               struct ompi_datatype_t *rdtype,
                                               struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module)
{
    int i;
    int communicator_size;
    size_t dsize, total_dsize;

    communicator_size = ompi_comm_size(comm);

    /* Special case for 2 processes */
    if (communicator_size == 2) {
        return ompi_coll_base_allgatherv_intra_two_procs(sbuf, scount, sdtype,
                                                         rbuf, rcounts, rdispls, rdtype,
                                                         comm, module);
    }

    /* Determine complete data size */
    if (MPI_IN_PLACE != sbuf) {
        ompi_datatype_type_size(sdtype, &dsize);
    } else {
        ompi_datatype_type_size(rdtype, &dsize);
    }

    total_dsize = 0;
    for (i = 0; i < communicator_size; i++) {
        total_dsize += dsize * (ptrdiff_t)rcounts[i];
    }

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_allgatherv_intra_dec_fixed"
                 " rank %d com_size %d msg_length %lu",
                 ompi_comm_rank(comm), communicator_size, (unsigned long)total_dsize));

    /* Decision based on allgather decision.   */
    if (total_dsize < 50000) {
        return ompi_coll_base_allgatherv_intra_bruck(sbuf, scount, sdtype,
                                                     rbuf, rcounts, rdispls, rdtype,
                                                     comm, module);
    } else {
        if (communicator_size % 2) {
            return ompi_coll_base_allgatherv_intra_ring(sbuf, scount, sdtype,
                                                        rbuf, rcounts, rdispls, rdtype,
                                                        comm, module);
        } else {
            return  ompi_coll_base_allgatherv_intra_neighborexchange(sbuf, scount, sdtype,
                                                                     rbuf, rcounts, rdispls, rdtype,
                                                                     comm, module);
        }
    }
}

/*
 *	gather_intra_dec
 *
 *	Function:	- seletects gather algorithm to use
 *	Accepts:	- same arguments as MPI_Gather()
 *	Returns:	- MPI_SUCCESS or error code, passed from corresponding
 *                        internal allgather function.
 */

int ompi_coll_tuned_gather_intra_dec_fixed(const void *sbuf, int scount,
                                           struct ompi_datatype_t *sdtype,
                                           void* rbuf, int rcount,
                                           struct ompi_datatype_t *rdtype,
                                           int root,
                                           struct ompi_communicator_t *comm,
                                           mca_coll_base_module_t *module)
{
    const int large_segment_size = 32768;
    const int small_segment_size = 1024;

    const size_t large_block_size = 92160;
    const size_t intermediate_block_size = 6000;
    const size_t small_block_size = 1024;

    const int large_communicator_size = 60;
    const int small_communicator_size = 10;

    int communicator_size, rank;
    size_t dsize, block_size;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_gather_intra_dec_fixed"));

    communicator_size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* Determine block size */
    if (rank == root) {
        ompi_datatype_type_size(rdtype, &dsize);
        block_size = dsize * (ptrdiff_t)rcount;
    } else {
        ompi_datatype_type_size(sdtype, &dsize);
        block_size = dsize * (ptrdiff_t)scount;
    }

    if (block_size > large_block_size) {
        return ompi_coll_base_gather_intra_linear_sync(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       root, comm, module,
                                                       large_segment_size);

    } else if (block_size > intermediate_block_size) {
        return ompi_coll_base_gather_intra_linear_sync(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       root, comm, module,
                                                       small_segment_size);

    } else if ((communicator_size > large_communicator_size) ||
               ((communicator_size > small_communicator_size) &&
                (block_size < small_block_size))) {
        return ompi_coll_base_gather_intra_binomial(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module);
    }
    /* Otherwise, use basic linear */
    return ompi_coll_base_gather_intra_basic_linear(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module);
}

/*
 *	scatter_intra_dec
 *
 *	Function:	- seletects scatter algorithm to use
 *	Accepts:	- same arguments as MPI_Scatter()
 *	Returns:	- MPI_SUCCESS or error code, passed from corresponding
 *                        internal allgather function.
 */

int ompi_coll_tuned_scatter_intra_dec_fixed(const void *sbuf, int scount,
                                            struct ompi_datatype_t *sdtype,
                                            void* rbuf, int rcount,
                                            struct ompi_datatype_t *rdtype,
                                            int root, struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    const size_t small_block_size = 300;
    const int small_comm_size = 10;
    int communicator_size, rank;
    size_t dsize, block_size;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_scatter_intra_dec_fixed"));

    communicator_size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    /* Determine block size */
    if (root == rank) {
        ompi_datatype_type_size(sdtype, &dsize);
        block_size = dsize * (ptrdiff_t)scount;
    } else {
        ompi_datatype_type_size(rdtype, &dsize);
        block_size = dsize * (ptrdiff_t)rcount;
    }

    if ((communicator_size > small_comm_size) &&
        (block_size < small_block_size)) {
        return ompi_coll_base_scatter_intra_binomial(sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     root, comm, module);
    }
    return ompi_coll_base_scatter_intra_basic_linear(sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     root, comm, module);
}
