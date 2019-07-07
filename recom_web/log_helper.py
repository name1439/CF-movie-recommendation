from recom_web import utils


def log(msg, type='INFO'):
    print(f'{utils.get_current_time()} {type}: {msg}')


def info(msg):
    log(msg)


def error(msg):
    log(msg, 'ERROR')


def warn(msg):
    log(msg, 'WARN')


# 主程序入口
if __name__ == '__main__':
    log('test log', 'INFO')
    warn('test info')
    pass
