import os
import sqlite3

from utils import NASConfig
from utils import get_logger

logger = get_logger('SQL')

sql_create_table = '''CREATE TABLE MODEL(
   ID INT PRIMARY KEY NOT NULL,
   STRUCTURE VARCHAR UNIQUE NOT NULL,
   TRAIN_TIME INT,
   LOSS DOUBLE,
   PREV_INDEX INT
   )
'''


def get_or_create_connection():
    if os.path.exists(NASConfig['SQL_FILE']):
        return sqlite3.connect(NASConfig['SQL_FILE'])
    conn = sqlite3.connect(NASConfig['SQL_FILE'])
    conn.execute(sql_create_table)
    conn.commit()
    return conn


def get_total_model_count():
    sql = '''
    SELECT COUNT(STRUCTURE) FROM MODEL
    '''
    conn = get_or_create_connection()
    r = conn.execute(sql).fetchall()
    conn.close()
    r = r[0][0]
    return r


def insert_new_model_config(id: int, structure: str,prev_index:int):
    sql = f'''
    INSERT INTO MODEL (ID, STRUCTURE, TRAIN_TIME,PREV_INDEX) VALUES ({id}, \"{structure}\", 0, {prev_index});
    '''
    conn = get_or_create_connection()
    conn.execute(sql)
    conn.commit()
    conn.close()
    # logger.info(sql)

def get_prev_record(structure = 'avgPool-3->avgPool-3->dense-96->maxPool-2->avgPool-2->avgPool-3->dense-48'):
    sql = f'''
    SELECT * FROM MODEL WHERE STRUCTURE = \'{structure}\'
    '''
    conn = get_or_create_connection()
    id, structure, train_time, loss, prev_index = conn.execute(sql).fetchall()[0]
    return id, structure, train_time, loss, prev_index


def insert_new_train_result(id: int, train_time: int, loss: float):
    sql = f'''
    UPDATE MODEL SET LOSS = {loss}, TRAIN_TIME = {train_time} WHERE ID = {id}
    '''
    try:
        conn = get_or_create_connection()
        conn.execute(sql)
        conn.commit()
        conn.close()
    except:
        logger.fatal(sql)