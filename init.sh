#!/usr/bin/env/sh

meteor='http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz'

mkdir -p ../bin ; cd ../bin; wget $meteor ; tar xvfz meteor-1.5.tar.gz ; rm meteor-1.5.tar.gz
