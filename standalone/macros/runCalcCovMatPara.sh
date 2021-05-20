#!/bin/sh
source /home/dolan/.bashrc
runnd280
echo "Input filename"
echo $1
echo "Variable"
echo $2
echo "Toy to process"
echo $3
echo "Work dir"
echo $4
cd $4
echo "I am here:"
pwd
root -b -l <<EOF
.L calcCovMat_fine_v2.C+
calcCovMat_fine_v2("$1", "tempOut.temp", "$2", 500, $3)
.q
EOF
#root -b -q -l "calcCovMat_fine_v2.C(\"$1\", \"tempOut.temp\", \"$2\", 500, $3)"