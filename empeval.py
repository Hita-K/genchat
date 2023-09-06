from rouge import Rouge
import textstat

reference = ["Yes, other genes besides APOE are associated with increased risk to develop dementia. However, these genes are relatively rare and typically found in families with a strong history of dementia and/or other related diseases. If you have questions or concerns about other genes increasing risk for dementia in your family, a neurology genetic counselor may assess your personal and family history and offer testing for additional genes, if indicated.", "Mild cognitive impairment, also called MCI, refers to a small but noticeable and measurable decline in cognition. Individuals with MCI may remain stable, return to aging normally, or progress to dementia. Dementia is a general term for a decline in mental ability severe enough to interfere with daily life. There are many different causes of dementia, but Alzheimer’s disease is the most common. Alzheimer’s disease is a progressive disease that causes decline in memory, thinking, and behavior over time.", "Each person inherits an APOE allele from each biological parent, meaning people can have one of six possible combinations: e2/e2, e2/e3, e2/e4, e3/e3, e3/e4, and e4/e4. Among these APOE alleles, e3 is the most common. About 60% of the population has the e3/e3 result. About 20-25% have at least one copy of e4 and 2-5% have two copies of e4. Only about 7% of individuals have the e2 allele.", "About 25% of the general population carries at least one copy of the e4 type of APOE.", "Alzheimer’s disease is considered “early-onset” when the disease develops before age 65.", "Everyone has some risk of developing Alzheimer’s disease.  The most common APOE result is e3/e3, which is associated with a 10-15% risk of developing Alzheimer’s disease by age 85. The e3 type of APOE is neutral, it does not increase or decrease the risk of Alzheimer’s disease.  When someone has one or two copies of the e4 type, the risk can increase.  The e3/e4 result has a 20-25% risk and the e4/e4 result has a 30-55% risk by age 85. The e2 type of APOE is more rare, which means we have less data about how it impacts the lifetime risk of Alzheimer’s disease.  The data we do have suggests that the e2 type of APOE reduces the risk for AD.  The risks for the e2/e2 and e2/e3 results may be slightly lower than the e3/e3 result, while the e2/e4 result might be slightly higher or slightly lower than the e3/e3 result. ", "Your children may or may not have the same APOE result as you.  This is because your children inherit one copy of APOE from you, and their second copy of APOE comes from their other parent.  For example, if you have two copies of the e3 type of APOE, you will always pass on one copy of e3 to your children.  But their final result will be determined by what type of APOE they inherit from their other parent. ", "Lifestyle changes that are healthy for the heart are generally considered healthy for brain aging. These lifestyle changes may include: exercise, eating a heart-healthy diet, reducing stress, and quitting smoking. It is also recommended to engage in social and mentally stimulating activities. Although none of these activities has been proven to prevent mild cognitive impairment or dementia due to Alzheimer’s disease, they may have other health benefits (such as reducing risk for cancer and heart disease).", "This is a difficult question to answer. While some occasional forgetfulness can be a normal part of aging, it would be important to consult with a doctor if you are noticing these difficulties occurring more often, or if they are interfering with normal daily life.", "When a brain is affected by Alzheimer’s, certain areas of the brain will be shrink (become smaller).  Doctors often call this shrinking “atrophy” and sometimes it can be seen using a brain scan (for example, a MRI).  If you were able to look inside the affected brain cells, you would see build-up of two specific proteins.  One of these proteins is called beta-amyloid and the other is called tau.  Some memory clinics are now able to detect these proteins using a special kind of brain scan, which helps doctors more accurately diagnosis Alzheimer’s disease."]

rouge = Rouge()

lines = ["Certainly. While the ApoE E4 allele is one of the most well-known genetic risk factors for Alzheimer's disease, there are other genes that have been identified as having associations with dementia. However, it's important to understand the context of this risk. The ApoE E4 gene is considered a major risk gene, but there are many other genes that are considered to have a smaller effect on risk. These are often called 'risk modifier genes.' Their individual impact on dementia risk is smaller compared to ApoE E4, but collectively, they might contribute. Nonetheless, the presence or absence of these genes offers only a piece of the puzzle. Dementia risk is influenced by a combination of genetic, environmental, and lifestyle factors. So, while genetics play a role, it's only part of the bigger picture of what determines dementia risk.", "Of course, it's crucial to understand the distinctions between these terms. At its core, dementia is a broad term that refers to a decline in cognitive abilities, affecting daily life. Within that spectrum, mild cognitive impairment (MCI) stands as a midway point between normal aging and dementia. It signifies noticeable cognitive changes, but not as severe as dementia. Not everyone with MCI will progress to dementia; some might even return to a stage of normal aging. Now, Alzheimer's disease is the most common cause of dementia. So, when we talk about Alzheimer's, we're referring to a specific type of dementia. To distill it: imagine dementia as an umbrella term under which conditions like Alzheimer's fit, and MCI is like a warning sign, indicating potential risk but not certainty.", "I understand that you're curious about the distribution of the ApoE gene variants in the general population. Recognizing that you already have some background knowledge on the topic, the E3 variant is the most common and is seen quite frequently in the population. On the other hand, the E2 variant is relatively rare but provides some protection, lowering the risk compared to the general population. The E4 variant, which you've tested for, falls on the higher end of the risk spectrum, but it's important to remember that not everyone with E4 variants will develop conditions like Alzheimer's. Many people share your curiosity and concern, and it's always good to relate your findings to a broader perspective to understand the bigger picture. Knowing these details can help you make more informed decisions about your health and lifestyle choices.", "I understand your concerns about the E4 type of the ApoE gene. While the E4 variant is associated with an increased risk of mild cognitive impairment and Alzheimer's disease, it's essential to note that it doesn't determine your destiny. Everyone carries two copies of the ApoE gene, one from each parent, and the E4 is just one of the three types—E2, E3, and E4. Some people have no copies of the E4 variant, others have one, and some, like you, have two. It's important to remember that approximately 33% of individuals with Alzheimer's disease don't have any copies of the E4 variant, indicating that other factors are involved. The presence of the E4 type increases the risk but does not guarantee the development of Alzheimer's. It's a piece of a larger puzzle, and there is still uncertainty about how it interacts with other factors to influence overall risk."]


# with open("qlist.txt", "r") as f:
#     lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()

    # ROUGE Scores
    scores = rouge.get_scores(line, reference[i])
    
    print(f"Line {i+1} ROUGE Scores:")
    for score_type, score_value in scores[0].items():
        print(f"  {score_type.upper()}:")
        for metric, value in score_value.items():
            print(f"    {metric}: {value}")


##########

# Readability

# with open("gen.txt", "r") as f:
#     ans = f.readlines()

def evaluate_readability(text):
    print(f"Text: {text}\n")
    
    fk = textstat.flesch_reading_ease(text)
    print(f"Flesch-Kincaid Reading Ease: {fk}")

    fk_grade = textstat.flesch_kincaid_grade(text)
    print(f"Flesch-Kincaid Grade Level: {fk_grade}")

    gf = textstat.gunning_fog(text)
    print(f"Gunning Fog Index: {gf}")

    cl = textstat.coleman_liau_index(text)
    print(f"Coleman-Liau Index: {cl}")


for i, line in enumerate(lines):
    # lines = lines.strip()
    print("question", i)
    evaluate_readability(lines[i])


# ###########

from langkit import textstat
from whylogs.experimental.core.udf_schema import udf_schema
import whylogs as why
text_schema = udf_schema()

profile = why.log({"input":"I like you. I love you."}, schema=text_schema).profile()

print(profile)