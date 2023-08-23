import os
import sys

## Replacement of os.chdir and os.system
class mos:
    def print(*args, sep=" "):
        print ("\u001b[31m", sep.join(map(str, args)), "\u001b[0m")

    def chdir(cmd):
        os.chdir(cmd)
        mos.print (">>> chdir:", cmd, "(cwd: %s)"%(os.getcwd()))

    def system(cmd, exit_on_error=True, wrapcmd="%s"):
        cmd = wrapcmd%(cmd)
        mos.print (">>>", cmd, "(cwd: %s)"%(os.getcwd()))
        ret = os.system(cmd)

        if exit_on_error and ret > 0:
            mos.print (">>> exec error:", ret)
            sys.exit(ret)
