/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/memchecker.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Send = PMPI_Send
#endif
#define MPI_Send PMPI_Send
#endif

static const char FUNC_NAME[] = "MPI_Send";

#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>          /* For ftruncate() */

#include "cuda_runtime.h"

static cudaStream_t d2h_stream = NULL;

static inline cudaStream_t get_stream() {
    if (NULL == d2h_stream) {
      cudaStreamCreate(&d2h_stream);
    }
    return d2h_stream;
}

static inline void* alloc_shared_mem(int count, int type_size, const char *name) {

  int fd = shm_open(name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );

  int mem_size = type_size * count;
  ftruncate(fd, mem_size);

  void* ptr = NULL;
  ptr = (void*) mmap(NULL, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (ptr==NULL) printf("mmap failed \n");
  return ptr;
}

unsigned long long test = 0;

int my_send(const void *buf, int count, struct ompi_datatype_t *type, int dest,
            int tag, mca_pml_base_send_mode_t mode,
            struct ompi_communicator_t* comm) {

    size_t type_size;
    ompi_datatype_type_size(type, &type_size);

    char str[256];
    sprintf(str, "%llu-%d-%d-%d", test, tag, dest, comm->c_my_rank);
    test++;
    void *ptr = alloc_shared_mem(count / 2, type_size, str);
    cudaMemcpyAsync(ptr, buf, ((count * type_size) / 2), cudaMemcpyDeviceToHost,
                    get_stream());


    MPI_Request request;
    int ret = MPI_SUCCESS;
    ret = MCA_PML_CALL(isend(buf, count / 2, type, dest, tag,
                                MCA_PML_BASE_SEND_STANDARD, comm, &request));
    ret = ompi_request_wait(&request, MPI_STATUS_IGNORE);
    cudaStreamWait(get_stream());
    return ret;
}

int MPI_Send(const void *buf, int count, MPI_Datatype type, int dest,
             int tag, MPI_Comm comm)
{
    int rc = MPI_SUCCESS;

    SPC_RECORD(OMPI_SPC_SEND, 1);

    MEMCHECKER(
        memchecker_datatype(type);
        memchecker_call(&opal_memchecker_base_isdefined, buf, count, type);
        memchecker_comm(comm);
    );

    if ( MPI_PARAM_CHECK ) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM, FUNC_NAME);
        } else if (count < 0) {
            rc = MPI_ERR_COUNT;
        } else if (tag < 0 || tag > mca_pml.pml_max_tag) {
            rc = MPI_ERR_TAG;
        } else if (ompi_comm_peer_invalid(comm, dest) &&
                   (MPI_PROC_NULL != dest)) {
            rc = MPI_ERR_RANK;
        } else {
            OMPI_CHECK_DATATYPE_FOR_SEND(rc, type, count);
            OMPI_CHECK_USER_BUFFER(rc, buf, type, count);
        }
        OMPI_ERRHANDLER_CHECK(rc, comm, rc, FUNC_NAME);
    }

    if (MPI_PROC_NULL == dest) {
        return MPI_SUCCESS;
    }

    OPAL_CR_ENTER_LIBRARY();
    rc = my_send(buf, count, type, dest, tag, MCA_PML_BASE_SEND_STANDARD, comm);
    OMPI_ERRHANDLER_RETURN(rc, comm, rc, FUNC_NAME);
}
