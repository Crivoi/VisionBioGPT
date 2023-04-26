from functools import cached_property

import psycopg2

from settings import POSTGRES_DB_NAME, POSTGRES_USERNAME, POSTGRES_PASSWORD


class MimicDatabase:
    """
    Representation of MIMIC-III local Postgres DB

    Notes:
        - identifies patients based on subject_id and hadm_id as a tuple
        - since we are only interested in using discharge summaries, it is important to note that there are no erroneous
          cases (where mimiciii.noteevents.iserror = '1'), thus it is not needed to filter for errors
    """

    def __init__(self):
        self.category = 'Discharge summary'

        self.conn = psycopg2.connect(
            database=POSTGRES_DB_NAME,
            user=POSTGRES_USERNAME,
            password=POSTGRES_PASSWORD,
            host='localhost',
            port='5432'
        )
        self.cursor = self.conn.cursor()

    @cached_property
    def top_icd9_codes(self):
        self.cursor.execute("""
            SELECT icd9_code FROM mimiciii.diagnoses_icd 
            GROUP BY icd9_code ORDER BY count(*) DESC LIMIT 50
            """)
        return [r[0] for r in self.cursor.fetchall()]

    @cached_property
    def count_discharge_summaries(self):
        """
        :return: Total number of discharge summaries linked to an icd9_code via subject_id and hadm_id
        """
        query = """
            SELECT count(*)
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
        """
        self.cursor.execute(query, [self.category])
        return self.cursor.fetchone()[0]

    @cached_property
    def count_hadm_ids(self):
        """
        :return: Number of distinct hospital admission ids
        """
        query = """
            SELECT count(distinct d.hadm_id)
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
        """
        self.cursor.execute(query, [self.category])
        return self.cursor.fetchone()[0]

    def query_text_and_icd9_code(self):
        query = """
            SELECT n.text, d.icd9_code
            FROM mimiciii.diagnoses_icd d 
            JOIN mimiciii.noteevents n
            ON d.subject_id = n.subject_id
            AND d.hadm_id = n.hadm_id
            AND n.category = %s
        """
        self.cursor.execute(query, [self.category])


if __name__ == '__main__':
    database = MimicDatabase()
    print(database.top_icd9_codes)
    print(database.count_discharge_summaries)
