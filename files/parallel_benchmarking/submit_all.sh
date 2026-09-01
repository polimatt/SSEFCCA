#!/bin/bash

for entry in ./*.job
do
  echo qsub "$entry"
done
