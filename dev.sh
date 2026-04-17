#!/bin/bash 

julia --threads=auto  --banner=no --project=test -i -e 'using Revise; include("test/setup.jl");'
