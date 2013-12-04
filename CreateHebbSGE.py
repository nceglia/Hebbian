## USAGE: python /home/baldig/projects/genomics/crick/crick/scripts/create_job_script.py variables hidden layers dropout rate crickjob

import sys
import os
import math
from HebbianNetwork import monotone_generator

content  = {}
variables = sys.argv[1]

count_vars = range(len(monotone_generator(variables)))
for i in range(len(count_vars)):
    count_vars[i] += 1
    count_vars[i] = str(count_vars[i])

content['len_iter_list'] = len(count_vars)
content['iter_list'] = "' '".join(count_vars)
#content['commands'] = sys.argv[2].replace("#", "${i}")
hidden = sys.argv[2]
layers = sys.argv[3]
dropout = sys.argv[4]
rate = sys.argv[5]

content['commands'] = "pypy /home/nceglia/codebase/Hebbian/HebbianNetwork.py --rate {0} --sigmoid 0 --examples 1000 --hidden {1} --variables {2} --layer {3} --rule oja --dropout {4} --table ".format(rate, hidden, variables, layers, dropout)+"${i}"
if len(sys.argv) > 3:
    content['name'] = sys.argv[-1].rstrip('.sge')
    try:
        os.mkdir('/extra/baldig3/projects/hebbian/qsub_dump/%s' % content['name'])
    except OSError:
        print "Cannot create output dir.."
else:
    content['name'] = "cjob"

template = open('js_hebb_template').read()

f = open(sys.argv[-1], 'w')
f.write(template % (content))
f.close()

print "Job Script created!"
