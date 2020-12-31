#ifndef HPC_MM_MT_HPP
#define HPC_MM_MT_HPP

#include <thread>
#include <hpc/matvec/gematrix.hpp>
#include <hpc/matvec/mm.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/aux/slices.hpp>

namespace hpc {

    template<
            template<typename> class MatrixA,
            template<typename> class MatrixB,
            template<typename> class MatrixC,
            typename T,
            typename Alpha,
            typename Beta,
            Require<
                    Ge<MatrixA<T>>,
                    Ge<MatrixB<T>>,
                    Ge<MatrixC<T>>
            > = true
    >
    void mm(Alpha alpha, const MatrixA<T> &A,
            const MatrixB<T> &B, Beta beta,
            MatrixC<T> &C,
            std::size_t nof_row_threads,
            std::size_t nof_col_threads) {
        assert(C.numRows() == A.numRows());
        assert(C.numCols() == B.numCols());
        assert(A.numCols() == B.numRows());

        const auto K = A.numCols();

        nof_row_threads = std::min(nof_row_threads, C.numRows());
        nof_col_threads = std::min(nof_col_threads, C.numCols());

        aux::UniformSlices rowSlices{nof_row_threads, C.numRows()};
        aux::UniformSlices colSlices{nof_col_threads, C.numCols()};

        std::vector<std::thread> threads;
        threads.reserve(nof_row_threads * nof_col_threads);

        for (auto r = 0U; r < nof_row_threads; ++r) {
            for (auto c = 0U; c < nof_col_threads; ++c) {
                threads.emplace_back([&, r, c](){
                    const auto rowOffset = rowSlices.offset(r);
                    const auto rowSize = rowSlices.size(r);
                    const auto colOffset = colSlices.offset(c);
                    const auto colSize = colSlices.size(c);

                    auto aBlock = A.block(rowOffset, 0).dim(rowSize, K);
                    auto bBlock = B.block(0, colOffset).dim(K, colSize);
                    auto cBlock = C.block(rowOffset, colOffset).dim(rowSize, colSize);

                    matvec::mm(alpha, aBlock, bBlock, beta, cBlock);
                });
            }
        }

        for (auto &thread : threads) {
            thread.join();
        }
    }
} // namespace hpc

#endif
