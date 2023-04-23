# get all patient notes and related ICD codes
"""
SELECT noteevents.text, diagnoses_icd.icd9_code
FROM mimiciii.diagnoses_icd NATURAL JOIN mimiciii.noteevents
"""
