from typing import Union, List

from nltk import RegexpTokenizer
from transformers import BioGptTokenizer

text = """Admission Date:  [**2174-2-12**]              Discharge Date:   [**2174-2-14**]

Date of Birth:  [**2122-4-28**]             Sex:   M

Service: MEDICINE

Allergies:
Patient recorded as having No Known Allergies to Drugs

Attending:[**First Name3 (LF) 2474**]
Chief Complaint:
shortness of breath

Major Surgical or Invasive Procedure:
None


History of Present Illness:
51 yo M with h/o asthma and right lung volume loss of unclear
etiology (?congenital hypoplasia), recurrent bronchitis in
winter, OSA, obesity, HTN who presented with 1 week of
productive cough, progresssive SOB, and over the past 2 days
weakness and fatigue to the point he was falling asleep at work.
He tried increasing his albuterol use but this did not help so
he came to the emergency room.

In the ED, initial vs were: T: 99.5 P109 BP101/56 R24 79% on RA
on presentation, 100% on NRB. The ED resident noted that he was
not particularly wheezy on exam. Labs notable for WBC of 5, HCT
34.5, sodium of 131 and creatinine of 1.0. Patient was given
ceftriaxone and azithromycin, nebs and 125 IV methylprednisone.
He had a CXR which was inconclusive, possible RML pneumonia.
Also be very tachycardic, EKG shows sinus tach. Got a CTA for
concern of PE-but no PE, but some increased interstitial
markings in RUL and RLL, more likely chronic process vs.
pneumonia. Tried to wean him down on 02, as soon as he would
fall asleep would desat. Right now, 40% venti mask and he is
[**Age over 90 **]%. Vitals on transfer, 99/5, HR 106, 127/87, 23.

On the floor, the patient notes that he feels fine while lying
still, but worse with movement. Denies sick contacts, recent
travel, new pets. Only recent med change is stopping zoloft 1
month ago. He worked previously in the printing business and
thinks he could have been exposed to some chemicals.

Review of systems:
(+) Per HPI
(-) Denies fever, night sweats, recent weight loss or gain.
Denies headache, sinus tenderness, rhinorrhea or congestion.
Denies chest pain, chest pressure, palpitations, or weakness.
Denies nausea, vomiting, diarrhea, constipation, abdominal pain,
or changes in bowel habits. Denies dysuria, frequency, or
urgency. Denies arthralgias or myalgias. Denies rashes or skin
changes.


Past Medical History:
Asthma
Hypertension
Hyperkalemia: intermittent elevations of his potassium.
Obesity
Glucose intolerance
Obstructive sleep apnea: declined CPAP therapy.
Anxiety
Vitamin B12 deficiency

Social History:
Married, no children. He works for [**Company 2475**]. Previously he worked
in a printing company where he reports that he was exposed to
fumes and did not wear a mask.
Tobacco:  Quit.
Alcohol:  Two glasses of wine per night and 3 bottles over the
weekend.
Drugs:  None.

Family History:
Grandmother with diabetes.  Father with Alzheimer's.


Physical Exam:
General: Alert, oriented, no acute distress
HEENT: Sclera anicteric, MMM, oropharynx clear
Neck: supple, JVP not elevated, no LAD
Lungs: Decreased breath sounds at R base, no wheezes, rales,
rhonchi
CV: Regular rate and rhythm, normal S1 + S2, no murmurs, rubs,
gallops
Abdomen:  soft, non-tender, non-distended, bowel sounds present,
no rebound tenderness or guarding, no organomegaly
GU: no foley
Ext: warm, well perfused, 2+ pulses, no clubbing, cyanosis or
edema

Pertinent Results:
[**2167-8-31**]
     Actual Pred %Pred Actual %Pred %chg
FVC   2.77  4.22   66    3.46   82 +25
FEV1  1.41  3.25   43    1.79   55 +27
MMF   0.59  3.53   17    0.62   18 +5
FEV1/FVC   51 77   66    51     66 +0
There is a moderate obstructive ventilatory defect with
  significant bronchdilator response.

Admission labs [**2174-2-12**]:

WBC-5.5 RBC-3.81* Hgb-11.7* Hct-34.8* MCV-91 MCH-30.8 MCHC-33.7
RDW-15.2 Plt Ct-336
PT-13.2 PTT-28.2 INR(PT)-1.1
Glucose-90 UreaN-18 Creat-1.0 Na-131* K-4.9 Cl-98 HCO3-21*
AnGap-17
Phos-5.0* Mg-1.9 Iron-34*
calTIBC-274 VitB12-405 Folate-10.2 Ferritn-249 TRF-211
Lactate-1.6

Discharge labs [**2174-2-14**]:

WBC-4.5 RBC-3.49* Hgb-10.9* Hct-33.4* MCV-96 MCH-31.1 MCHC-32.5
RDW-15.1 Plt Ct-307
Glucose-86 UreaN-22* Creat-1.1 Na-137 K-4.5 Cl-105 HCO3-23
AnGap-14

Micro:
[**2-12**] Blood cultures- pending at time of discharge (negative to
date)
[**2-12**] MRSA screen pending
[**2-12**] Urine culture and legionella- negative
[**2-12**] RSV screen negative

Imaging:
[**2-12**] EKG: Sinus tachycardia. Right bundle-branch block. No
previous tracing available for comparison.

[**2-12**] CXR:
SINGLE FRONTAL PORTABLE VIEW OF THE CHEST: As compared to prior
study there is minimal increase in right mid - lower lung
opacity . Right lower lobe pleural thickening is grossly stable.
Bullous changes denoted by relative lucency in the right upper
lobe are again noted and unchanged. Prominent right mediastinal
- paratracheal soft tissue remains unchanged. Left lung is
clear. Heart is not enlarged. The aortic contour is grossly
unremarkable.

IMPRESSION: Minimal interval increase in opacification of the
right mid -
lower lung is better evaluated on subsequent CT from same day.

[**2-12**] CTA Chest:
1. No pulmonary embolus.
2. Right lung volume loss could be due to a congenital anomaly
or pulmonary infection early in life.
3. Persistent area of round atelectasis in the right lower lobe
with
increased size slightly since [**2166**].
4. New subpleural septal thickening in the left upper lobe and
medial basal left lower lobe, new since [**2166**] and suggestive of
an interstitial fibrosis.
5. Mildly enlarged 13-mm prevascular lymph node.

Given the pulmonary interstitial findings and mildly enlarged
lymph node,
followup high-resolution CT is suggested, preferrably in 3
months if not
needed earlier.

[**2-14**] Transthoracic echocardiogram:
The left atrium and right atrium are normal in cavity size. The
right atrial pressure is indeterminate. Mild symmetric left
ventricular hypertrophy with normal cavity size, and global
systolic function (LVEF>55%). Due to suboptimal technical
quality, a focal wall motion abnormality cannot be fully
excluded. The right ventricular cavity is moderately dilated
with moderate global free wall hypokinesis. There is abnormal
systolic septal motion/position consistent with right
ventricular pressure overload. The aortic valve leaflets are
mildly thickened (?#). There is no aortic valve stenosis. No
aortic regurgitation is seen. The mitral valve appears
structurally normal with trivial mitral regurgitation. There is
moderate pulmonary artery systolic hypertension. There is no
pericardial effusion.

IMPRESSION: Suboptimal image quality. Right ventricular cavity
enlargement with free wall hypokinesis and moderate pulmonary
artery systolic hypertension. Mild symmetric left ventricular
hypertrophy with normal cavity size and global systolic
function.
This constellation of findings is suggestive of a primary
pulmonary process, e.g., pulmonary embolism.

Brief Hospital Course:
In the ED, initial vs were: T: 99.5 P109 BP101/56 R24 79% on RA
on presentation, 100% on NRB. Labs notable for WBC of 5, HCT
34.5, sodium of 131 and creatinine of 1.0. Patient was given
ceftriaxone and azithromycin, nebs, and methylprednisone.  He
had a CXR which was possibly suggestive of RML pneumonia, and a
CTA that showed no signs of PE but did reveal some increased
interstitial markings in the RUL and RLL, more consistent with a
chronic process vs. pneumonia. the patient was admitted to the
ICU for monitoring on [**2-12**]. He had in past refused CPAP for his
OSA but accepted it and reported he had an unusually restful
night. He was transferred to the medicine floor on [**2-13**].

#  Hypoxia: His increased cough and the CT and chest Xray
imaging were most consistent with a right middle lobe pneumonia
with underlying chronic pneumonic process, and he will need
another high-resolution chest CT in 3 months to evaluate
progression of these findings. In the ICU, he continued
antibiotic treatment for CAP. He did not receive further
steroids. He had an Echo that showed RV enlargement, free wall
hypokinesis & moderate elevation of PASP. Although these
findings were consistent with possible pulmonary embolism, he
had a CTA on admission that was negative for PE. Pulmonary was
consulted and felt his right heart findings could be related to
his underlying known chronic lung disease and sleep apnea rather
than PE. After discussion with his PCP and pulmonologist, he was
discharged to have a repeat sleep study performed and follow-up
TTE in 1 month.  Patient was encouraged to start using home
CPAP.

# Hyponatremia: He was noted to be mildly hyponatremic at
admission, possibly due to salt-wasting from his HCTZ use. This
mild hyponatremia resolved with a liter of normal saline.

# Obstructive sleep apnea: He is now agreeable to trialing CPAP.
He was told he should have an outpatient sleep study as above.

# Normocytic Anemia: His hematocrit was slightly lower than
recent baseline, and iron studies were consistent with chronic
inflammation.  This will be further worked up as an outpatient.

Medications on Admission:
# Albuterol Sulfate [ProAir HFA] 90 mcg HFA Aerosol Inhaler 2
puffs inhaled every 4 hr as needed for asthma
# Amlodipine 5 mg Tablet 1 (One) Tablet(s) by mouth once a day
# Fluticasone-Salmeterol [Advair Diskus] 500 mcg-50 mcg/Dose
Disk with Device 1 (One) puff twice a day
# Hydrochlorothiazide 25 mg Tablet 1 Tablet(s) by mouth daily

# Lisinopril 40 mg Tablet 1 Tablet(s) by mouth once a day
# Tiotropium Bromide [Spiriva with HandiHaler] 18 mcg Capsule,
w/Inhalation Device one diskus inhaled once a day

Discharge Medications:
1. Albuterol Sulfate 90 mcg/Actuation HFA Aerosol Inhaler Sig:
Two (2) puffs Inhalation every four (4) hours as needed for
shortness of breath or wheezing.
2. Hydrochlorothiazide 25 mg Tablet Sig: One (1) Tablet PO once
a day.
3. Amlodipine 5 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).

4. Advair Diskus 500-50 mcg/Dose Disk with Device Sig: One (1)
puff Inhalation every twelve (12) hours.
5. Lisinopril 40 mg Tablet Sig: One (1) Tablet PO once a day.
6. Spiriva with HandiHaler 18 mcg Capsule, w/Inhalation Device
Sig: One (1) diskus Inhalation once a day.
7. Levofloxacin 750 mg Tablet Sig: One (1) Tablet PO once a day
for 4 days.
Disp:*4 Tablet(s)* Refills:*0*


Discharge Disposition:
Home

Discharge Diagnosis:
Primary:
Hypoxia
Community acquired pneumonia
Obstructive sleep apnea

Secondary:
Asthma
Hypertension


Discharge Condition:
Mental Status: Clear and coherent
Level of Consciousness: Alert and interactive
Activity Status: Ambulatory - Independent


Discharge Instructions:
You came to the hospital with shortness of breath and fatigue.
You were briefly in the intensive care unit for close monitoring
and your breathing improved with antibiotics and steroids.  You
were found to have pneumonia and are being treated with
antibiotics.  You had an echocardiogram of your heart that
showed some strain on the right side of your heart, and this is
likely due to your lung disease.  Please follow-up with your PCP
[**Last Name (NamePattern4) **]. [**First Name (STitle) 1022**] and your pulmonologist Dr. [**Last Name (STitle) **] as below.  You have
also been scheduled to see Dr. [**First Name (STitle) 437**] in sleep medicine to
discuss your sleep apnea.

The following changes were made to your medications:
Started levofloxacin, an antibiotic, once daily.  You should
take this for 4 more days.

Followup Instructions:
Please follow-up with your PCP, [**Last Name (NamePattern4) **]. [**First Name (STitle) 1022**], on [**2-21**] at 4:20pm.  You
should have another echocardiogram done in 1 month to evaluate
your heart function.

You have an appointment to see Dr. [**First Name (STitle) 437**], a sleep physician, [**Name10 (NameIs) **]
[**2-22**] at 8 AM, [**Hospital Ward Name 23**] building [**Location (un) **] to discuss your
sleep apnea. You should be fitted for a CPAP machine at home to
treat your obstructive sleep apnea.  If you need to reschedule,
you can call [**Telephone/Fax (1) 612**].

Please follow-up with Pulmonary (Dr. [**Last Name (STitle) **] on [**3-22**] at
8:45 AM on the [**Location (un) 436**] of the [**Hospital Ward Name 23**] building, [**Location (un) 2476**].  If an earlier appointment is available, they will contact
you.  If you need to reschedule your appointment, please call
[**Telephone/Fax (1) 612**].

A CT of the chest showed subpleural septal thickening in the
left upper and medial basal lower lobes, new since [**2166**], and a
mildly enlarged 13-mm prevascular lymph node. Given these
findings, you should have a second high-resolution CT in the
next 3 months to determine whether these changes resolve on
their own.


                             [**First Name11 (Name Pattern1) **] [**Last Name (NamePattern4) 2477**] MD, [**MD Number(3) 2478**]

"""


class Preprocessor:
    regexp_tokenizer = RegexpTokenizer(r'\w+')
    biogpt_tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')

    @classmethod
    def tokenize(cls, text: str):
        return cls.regexp_tokenizer.tokenize(text.lower())

    @classmethod
    def encode(cls, text: Union[List[str], str]):
        return cls.biogpt_tokenizer.encode(text)


def sliding_window(input_array, window_size=512, stride=256):
    output_arrays = []
    for i in range(0, len(input_array), stride):
        window = input_array[i:i + window_size]
        if len(window) == window_size:
            output_arrays.append(window)
        elif len(window) > 0:
            output_arrays.append(window)
    return output_arrays


if __name__ == '__main__':
    pass
