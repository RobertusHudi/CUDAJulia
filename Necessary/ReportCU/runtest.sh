#!/bin/sh
for size in 8192 12288 16384 20480
do
	#cuda-memcheck ./mmul $size > "memcheck_$size.txt"
	nsys profile -o "report$size" --stats=true ./mmNaive4 $size > "result$size.txt"
done