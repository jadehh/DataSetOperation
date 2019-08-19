#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/15 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/15  下午3:44 modify by jade

import os
import shutil
import sys
def install():
    for sys_path in sys.path:
        if "site-packages" in sys_path and os.path.isdir(sys_path):
            if os.path.exists(os.path.join(sys_path, "datasetopeation/")):
                shutil.rmtree(os.path.join(sys_path, "datasetopeation/"))
            shutil.copytree("datasetopeation/", os.path.join(sys_path, "datasetopeation/"))
            print ("Install to "+sys_path)
            break



if __name__ == '__main__':
    install()