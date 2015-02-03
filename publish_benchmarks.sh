#!/bin/bash
CONDAROOT=$(conda info -s | grep sys.prefix | awk '{print $2}')
PYTHONPATH=$CONDAROOT/lib/python3.4/site-packages:$CONDAROOT/lib/python3.4/site-packages/Mako-1.0.0-py3.4.egg asv publish
