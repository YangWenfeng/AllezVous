#!/bin/sh
cd "$( dirname $0 )"
echo "http://localhost:8337/"
python -m SimpleHTTPServer 8337 

