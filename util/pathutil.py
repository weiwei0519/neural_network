# coding=UTF-8
# 获取项目路径

import sys
import os


class PathUtil(object):
    """路径处理工具类"""

    def __init__(self):
        # 判断调试模式
        debug_vars = dict((a, b) for a, b in os.environ.items() if a.find('IPYTHONENABLE') >= 0)
        # 根据不同场景获取根目录
        if len(debug_vars) > 0:
            """当前为debug运行时"""
            self.rootPath = sys.path[2]
        elif getattr(sys, 'frozen', False):
            """当前为exe运行时"""
            self.rootPath = os.getcwd()
        else:
            """正常执行"""
            self.rootPath = sys.path[1]
        # 替换斜杠
        self.rootPath = self.rootPath.replace("\\", "/")

    def getPathFromResources(self, fileName):
        """按照文件名拼接资源文件路径"""
        filePath = "%s/resources/%s" % (self.rootPath, fileName)
        return filePath


if __name__ == '__main__':
    propath = PathUtil()
    """测试"""
    # path = PathUtil.getPathFromResources("context.ini")
    print(propath.rootPath)
