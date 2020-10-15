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
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/memchecker.h"
#include "ompi/request/request.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Recv = PMPI_Recv
#endif
#define MPI_Recv PMPI_Recv
#endif

static const char FUNC_NAME[] = "MPI_Recv";

#include "cuda_runtime.h"

void *host_mem = NULL;
static cudaStream_t d2h_stream = NULL;

static inline void* get_host_mem() {
    if (NULL == host_mem) {
        size_t mem_size = 1024 * 1024 * 1024;
        cudaMallocHost(&host_mem, mem_size);
    }
    return host_mem;
}

static inline cudaStream_t get_stream() {
    if (NULL == d2h_stream) {
      cudaStreamCreate(&d2h_stream);
    }
    return d2h_stream;
}
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>          /* For ftruncate() */

static inline void* alloc_shared_mem(int count, int type_size, const char *name) {

  int fd = shm_open(name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );

  int mem_size = type_size * count;
  ftruncate(fd, mem_size);

  void* ptr = NULL;
  ptr = (void*) mmap(NULL, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (ptr==NULL) printf("mmap failed \n");
  return ptr;
}
unsigned long long rtest = 0;

int my_recv(void *buf, size_t count, struct ompi_datatype_t *type, int src,
            int tag, struct ompi_communicator_t* comm,
            ompi_status_public_t* status) {
    MPI_Request request;
    int ret = MPI_SUCCESS;
    ret = MCA_PML_CALL(irecv(buf, count / 2, type, src, tag, comm, &request));

    size_t type_size;
    ompi_datatype_type_size(type, &type_size);

    char str[256];
    sprintf(str, "%llu-%d-%d-%d", rtest, tag, comm->c_my_rank, src);

    void *ptr = alloc_shared_mem(count / 2, type_size, str);
    cudaMemcpyAsync(buf, ptr, ((count * type_size) / 2), cudaMemcpyHostToDevice,
                    get_stream());

    ret = ompi_request_wait(&request, status);
    cudaStreamWait(get_stream());
    return ret;
}

int MPI_Recv(void *buf, int count, MPI_Datatype type, int source,
             int tag, MPI_Comm comm, MPI_Status *status)
{
    int rc = MPI_SUCCESS;

    SPC_RECORD(OMPI_SPC_RECV, 1);

    MEMCHECKER(
        memchecker_datatype(type);
        memchecker_call(&opal_memchecker_base_isaddressable, buf, count, type);
        memchecker_comm(comm);
    );

    if ( MPI_PARAM_CHECK ) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        OMPI_CHECK_DATATYPE_FOR_RECV(rc, type, count);
        OMPI_CHECK_USER_BUFFER(rc, buf, type, count);

        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM, FUNC_NAME);
        } else if (((tag < 0) && (tag != MPI_ANY_TAG)) || (tag > mca_pml.pml_max_tag)) {
            rc = MPI_ERR_TAG;
        } else if ((source != MPI_ANY_SOURCE) &&
                   (MPI_PROC_NULL != source) &&
                   ompi_comm_peer_invalid(comm, source)) {
            rc = MPI_ERR_RANK;
        }

        OMPI_ERRHANDLER_CHECK(rc, comm, rc, FUNC_NAME);
    }

    if (MPI_PROC_NULL == source) {
        if (MPI_STATUS_IGNORE != status) {
            *status = ompi_request_empty.req_status;
        }
        return MPI_SUCCESS;
    }

    OPAL_CR_ENTER_LIBRARY();

    rc = my_recv(buf, count, type, source, tag, comm, status);
    OMPI_ERRHANDLER_RETURN(rc, comm, rc, FUNC_NAME);
}
