import os
import psycopg2
from utils.Connection import all_tables
from utils.Connection import conn
TIME_END = 0.25


class PostgreEnv(object):
    def __init__(self, tables):
        self.score = 0.0
        self.tables = tables
        self.state = 0
        self.collection = []
        # self.space 
        self.time_start = time.time()

    def step(self, action):
        #:0: dict index 1:join type 2:join order
        #vector 当前state encoder情况
        #state: 0 未完成，1已完成，2超时
        if time.time() - time_start > TIME_END:
            self.state = 2
        elif len(self.tables) == 0:
            self.state = 1
        else:
            self.state = 0

    def eval(self, sql):
        print(1)

