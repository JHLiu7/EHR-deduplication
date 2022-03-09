with adm_base as (
    select 
        SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, HOSPITAL_EXPIRE_FLAG, ADMISSION_TYPE,
        ROW_NUMBER() OVER(PARTITION BY SUBJECT_ID order by ADMITTIME) as rnum   
    from `physionet-data.mimiciii_clinical.admissions` 
    order by SUBJECT_ID, HADM_ID, ADMITTIME
), admission_first as (
    select *
    from adm_base 
    where 
        rnum = 1 and (HOSPITAL_EXPIRE_FLAG=0)
), admission_second_tmp as (
    select
        SUBJECT_ID, HADM_ID, ADMITTIME as readtime
    from adm_base 
    where
        rnum > 1 
        # and ADMISSION_TYPE !='ELECTIVE' # remove elective adm
), admission_second as (
    select
        SUBJECT_ID, min(ad.readtime) READMITTIME
    from
        admission_second_tmp ad 
    group by
        SUBJECT_ID
), admission as (
    select 
        a1.*, a2.READMITTIME
    from 
        admission_first a1 left join admission_second a2 on a1.SUBJECT_ID=a2.SUBJECT_ID
), age as (
    select 
        t.HADM_ID, DATETIME_DIFF(t.ADMITTIME, p.DOB, DAY) / 365 age
    from
        admission t 
    left join `physionet-data.mimiciii_clinical.patients` p on t.SUBJECT_ID = p.SUBJECT_ID
), note_crit as (
    select 
        distinct HADM_ID
    from 
        `physionet-data.mimiciii_notes.noteevents`
    where
        CATEGORY != 'Discharge summary'
)

select 
    ad.SUBJECT_ID, ad.HADM_ID, ad.DISCHTIME, ad.READMITTIME, 
    case 
        when DATETIME_DIFF(READMITTIME, DISCHTIME, HOUR) < 30*24 then 1 
        else 0
    end as READM_30
from admission ad
    inner join note_crit n on ad.HADM_ID = n.HADM_ID
    left join age a on ad.HADM_ID = a.HADM_ID
where a.age >=15


# 34028 queried on 10/18/2021

