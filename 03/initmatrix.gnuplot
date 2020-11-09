set terminal svg size 900, 500
set output "initmatrix.svg"
set xlabel "Matrix dim A: M=N"
set ylabel "Time"
set title "matrix initialization"
set key outside
set pointsize Time0.5
plot "debug_gcc.data" using 1:3 with linespoints lt 2 lw 3 title "col-major (gcc, debug)", \
     "debug_gcc.data" using 1:4 with linespoints lt 3 lw 3 title "row-major (gcc, debug)", \
     "release_gcc.data" using 1:3 with linespoints lt 4 lw 3 title "col-major (gcc, release)", \
     "release_gcc.data" using 1:4 with linespoints lt 5 lw 3 title "row-major (gcc, release)", \
     "debug_clang.data" using 1:3 with linespoints lt 6 lw 3 title "col-major (clang, debug)", \
     "debug_clang.data" using 1:4 with linespoints lt 7 lw 3 title "row-major (clang, debug)", \
     "release_clang.data" using 1:3 with linespoints lt 8 lw 3 title "col-major (clang, release)", \
     "release_clang.data" using 1:4 with linespoints lt 9 lw 3 title "row-major (clang, release)"
