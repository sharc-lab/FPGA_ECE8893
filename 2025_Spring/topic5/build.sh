#!/bin/bash

CC=gcc


rm 16 32 64 2>/dev/null

$CC -O3 -DBIT16 test.c -o 16 || {
    echo "cannot build 16-bit binary"
    exit 1
}

$CC -O3 -DBIT32 test.c -o 32 || {
    echo "cannot build 32-bit binary"
    exit 1
}
$CC -O3  test.c -o 64 || {
    echo "cannot build 64-bit binary"
    exit 1
}


echo 
timeout -s 9 60 ./16

echo 
timeout -s 9 60 ./32

echo 
timeout -s 9 60 ./64
