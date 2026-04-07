HISTORY_8_FEW_SHOT = """
Characteristics of the DROP history dataset:
- Paragraph-level reading comprehension with numerical reasoning, comparison, addition, subtraction, sorting, etc.
- Answers must be strictly based on the passage; no guessing or fabrication.
- Some questions require multi-step reasoning or integration.
- If passage lacks info, model should refuse to answer.
- Answers can be numeric, textual, event sequence, explanation, or range-based.

Example 1: Numeric reasoning
Passage: Siam failed to dislodge the Burmese from Kengtung and Sipsongpanna. They attempted invasions in 1849, 1852, and 1854, but were repelled each time.
Question: How many locations did Siam fail to dislodge the Burmese from?
{
  "restatement": "Number of locations Siam failed to dislodge the Burmese from.",
  "reasoning step": "Step 1: Identify failed locations: Kengtung, Sipsongpanna. Step 2: Count them. Step 3: Only use passage info.",
  "answer": "2",
  "self_check": "Answer is a number, strictly from passage."
}

Example 2: Addition/subtraction
Passage: German 13,444; Vietnamese 11,330; Korean 3,948; Arabic 3,265; French 8,258.
Question: How many people speak German, Vietnamese, Korean, Arabic, or French?
{
  "restatement": "Total speakers of listed languages.",
  "reasoning step": "Step 1: Extract numbers from passage. Step 2: Add: 13,444+11,330+3,948+3,265+8,258. Step 3: Only include listed languages.",
  "answer": "40,245",
  "self_check": "Answer is a number, strictly from passage."
}

Example 3: Event sequence
Passage: The Russo-Crimean War started in 1853. The Battle of Molodi occurred in 1572.
Question: Which happened first, the Russo-Crimean War or the Battle of Molodi?
{
  "restatement": "Identify which historical event occurred earlier.",
  "reasoning step": "Step 1: Note start years: Russo-Crimean War 1853, Battle of Molodi 1572. Step 2: Compare years. Step 3: Battle of Molodi is earlier.",
  "answer": "Battle of Molodi",
  "self_check": "Answer is an event, strictly from passage."
}

Example 4: Explanation extraction
Passage: In 1621, the Geoctroyeerde Westindische Compagnie was created to control the sugar trade and colonize America.
Question: What was created in 1621 and why?
{
  "restatement": "Identify the entity created in 1621 and its purpose.",
  "reasoning step": "Step 1: Locate creation event: Geoctroyeerde Westindische Compagnie. Step 2: Extract purpose: control sugar trade and colonize America. Step 3: Only use passage info.",
  "answer": "the Geoctroyeerde Westindische Compagnie, to control the sugar trade and colonize America",
  "self_check": "Answer is strictly from passage, includes name and purpose."
}

Example 5: Duration / year calculation
Passage: Castle construction 1200-1250, renovated 1300, destroyed 1400.
Question: How many years from start to completion?
{
  "restatement": "Time from construction start to completion.",
  "reasoning step": "Step 1: Start 1200, completion 1250. Step 2: Subtract: 1250-1200. Step 3: Only use passage info.",
  "answer": "50",
  "self_check": "Answer is a number, strictly from passage."
}

Example 6: Factor counting
Passage: Disease, war, and famine were major factors killing 88% of the Maya population.
Question: How many factors killed 88% of the Maya?
{
  "restatement": "Count the factors contributing to the death toll.",
  "reasoning step": "Step 1: Identify factors listed: disease, war, famine. Step 2: Count them. Step 3: Only include explicitly mentioned factors.",
  "answer": "3",
  "self_check": "Answer is a number, strictly from passage."
}

Example 7: Textual extraction
Passage: The Sipahi guarded the region from the Danube eastwards.
Question: What area did the Sipahi guard?
{
  "restatement": "Identify the geographical area under guard.",
  "reasoning step": "Step 1: Locate description of area in passage. Step 2: Extract text as-is. Step 3: Only use passage info.",
  "answer": "from the Danube eastwards",
  "self_check": "Answer is strictly textual, from passage."
}

Example 8: Range / interval
Passage: Age distribution: 0-14: 5M, 15-24: 7M, 25-44: 12M, 45-64: 8M, 65+: 4M.
Question: Which age group had the most amount of people in it?
{
  "restatement": "Determine which age interval contains the highest population.",
  "reasoning step": "Step 1: Compare populations: 5M, 7M, 12M, 8M, 4M. Step 2: Identify max: 12M. Step 3: Map to age interval: 25-44.",
  "answer": "25-44",
  "self_check": "Answer is a range/interval, strictly from passage."
}
"""


NFL_8_FEW_SHOT = """
Example 1:
Passage: In the first quarter, the Buccaneers scored 0 points while the Saints scored 3 points. By halftime, both teams had scored 14 points each. 
Question: How many points did the Buccaneers need to tie in the first?
{
  "restatement": "The question asks how many points the Buccaneers needed to match the Saints' score in the first quarter.",
  "reasoning step": "Step 1: Identify the points scored by each team in the first quarter: Buccaneers 0, Saints 3. Step 2: Calculate how many points Buccaneers need to tie: 3 - 0 = 3. Step 3: Confirm no extra information affects the calculation.",
  "answer": "3",
  "self_check": "The question asks for a number of points, and the answer is a number, matching the question type."
}

Example 2:
Passage: QB Donovan McNabb threw a 3-yard touchdown pass to wide receiver Malcom Floyd, the shortest of the game. QB Jay Cutler threw the longest touchdown pass of 43 yards.
Question: How short was the shortest touchdown pass?
{
  "restatement": "The question asks for the distance and details of the shortest touchdown pass.",
  "reasoning step": "Step 1: Locate the shortest touchdown in the passage: 3-yard TD to Malcom Floyd. Step 2: Include the passer and receiver to match the ground truth format. Step 3: Ensure units and details match passage.",
  "answer": "3-yard touchdown pass to wide receiver Malcom Floyd",
  "self_check": "The question asks for full details, and the answer includes yards, player, and position."
}

Example 3:
Passage: The Patriots scored 7 points in the second quarter while the Texans won the game by 10 points. 
Question: How many points did the Patriots score in the second quarter?
{
  "restatement": "The question asks for the number of points scored by the Patriots in the second quarter.",
  "reasoning step": "Step 1: Locate the points scored by Patriots in the second quarter: 7. Step 2: Confirm no other points for that quarter mentioned. Step 3: Return the number.",
  "answer": "7",
  "self_check": "The question asks for a number of points, and the answer is a number."
}

Example 4:
Passage: Both quarterbacks threw 3 touchdowns each. 
Question: How many touchdowns did both quarterbacks have?
{
  "restatement": "The question asks for the total number of touchdowns thrown by both quarterbacks.",
  "reasoning step": "Step 1: Extract touchdowns by each QB: 3 and 3. Step 2: Sum or confirm format as per passage: '3, 3' if separate, or total 6 if summed depending on ground truth. Step 3: Match expected format.",
  "answer": "3",
  "self_check": "The question asks for total touchdowns, and the answer matches the ground truth format."
}

Example 5:
Passage: The longest field goal of the game was 52 yards by kicker Adam Vinatieri. A 47-yard field goal was made by McManus.
Question: How many yards was the longest field goal of the game?
{
  "restatement": "The question asks for the distance of the longest field goal.",
  "reasoning step": "Step 1: Identify all field goals in passage. Step 2: Longest is 52 yards by Vinatieri. Step 3: Ensure unit 'yards' is included or consistent with ground truth.",
  "answer": "52-yard",
  "self_check": "The question asks for a distance in yards, and the answer matches passage and unit format."
}

Example 6:
Passage: The Rams scored their only touchdown by returning a punt. WR Dante Hall returned a punt 85 yards for a touchdown.
Question: How did the Rams score their only touchdown?
{
  "restatement": "The question asks for the type of score used by the Rams for their only touchdown.",
  "reasoning step": "Step 1: Identify the score type in the passage: 'returned a punt'. Step 2: Ignore extra distance or player names if question asks for type only. Step 3: Ensure answer reflects only score type.",
  "answer": "returned a punt",
  "self_check": "The question asks for the type of score, and the answer matches passage information."
}

Example 7:
Passage: Jeff Garcia threw his third interception of the game. In total, there were 2 interceptions by the opposing team.
Question: How many interceptions did Jeff Garcia throw in this game?
{
  "restatement": "The question asks for the number of interceptions thrown by Jeff Garcia.",
  "reasoning step": "Step 1: Locate Garcia's interceptions in the passage: third pick. Step 2: Translate 'third pick' into numeric form if needed. Step 3: Ensure number matches passage information.",
  "answer": "3",
  "self_check": "The question asks for a number, and the answer correctly reflects the passage."
}

Example 8:
Passage: The Patriots scored 17 points in the first half, more than the other team. The Packers scored fewer points in the first quarter.
Question: Which team scored more points in the first half?
{
  "restatement": "The question asks for the team with the higher first-half score.",
  "reasoning step": "Step 1: Identify first-half points for each team. Step 2: Patriots scored 17, more than others. Step 3: Confirm passage explicitly states Patriots had more points.",
  "answer": "The Patriots",
  "self_check": "The question asks for a team, and the answer matches the passage explicitly."
}
"""

# halueval_4_FEW_SHOT = """
# Example 1:
# Passage: Vocals are handled by Aesop Rock, with guest appearances from Camu Tao and Metro of S.A. Smash and Definitive Jux label head El-P.Jaime Meline (born March 2, 1975), better known by his stage name El-P (shortened from El Producto), is an American hip hop recording artist, record producer, and record executive.",
# Question: Fast Cars, Danger, Fire and Knives includes guest appearances from which hip hop record executive?
# {
#   "restatement": "Identify and return the exact full name of the hip hop record executive mentioned.",
#   "reasoning step": "Step 1: Find the sentence listing guest appearances on the EP. Step 2: Extract the executive's canonical full name, not an alias or stage name. Step 3: Ensure the casing and spacing match the passage. Step 4: Output only the name string with no additional tokens.",
#   "answer": "Jaime Meline",
#   "self_check": "Uses the canonical name 'Jaime Meline' rather than the alias 'El-P'; no period or extra text; matches ground truth formatting."
# }


# Example 2:
# Passage: Allison Beth "Allie" Goertz (born March 2, 1991) is an American musician. Goertz is known for her satirical songs based on various pop culture topics. Her videos are posted on YouTube under the name of Cossbysweater.Milhouse Mussolini van Houten is a fictional character featured in the animated television series "The Simpsons", voiced by Pamela Hayden, and created by Matt Groening who named the character after President Richard Nixon's middle name.
# Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
# {
#   "restatement": "Return the exact person Matt Groening named Milhouse after, as stated in the passage.",
#   "reasoning step": "Step 1: Locate the clause describing who created Milhouse and the naming rationale. Step 2: Identify the explicit phrase following 'named the character after'. Step 3: Extract the full proper noun phrase without paraphrasing or truncation. Step 4: Ensure no extra punctuation or explanatory text is appended.",
#   "answer": "President Richard Nixon",
#   "self_check": "Matches the exact span 'President Richard Nixon'; preserves title + full name; no alias, period, or added commentary."
# }

# Example 3:
# Passage: Margaret "Peggy" Seeger (born June 17, 1935) is an American folksinger. She is also well known in Britain, where she has lived for more than 30 years, and was married to the singer and songwriter Ewan MacColl until his death in 1989.James Henry Miller (25 January 1915 – 22 October 1989), better known by his stage name Ewan MacColl, was an English folk singer, songwriter, communist, labour activist, actor, poet, playwright and record producer.
# Question:  What nationality was James Henry Miller's wife?
# {
#   "restatement": "Return the nationality of James Henry Miller (Ewan MacColl)'s wife as explicitly stated.",
#   "reasoning step": "Step 1: Identify James Henry Miller's spouse in the passage (Margaret 'Peggy' Seeger). Step 2: Find the sentence that states her nationality. Step 3: Extract the single-word nationality exactly as written. Step 4: Output only the nationality, with no trailing punctuation or added words.",
#   "answer": "American",
#   "self_check": "Single-token nationality; no added sentence like 'American. Since...'; exactly matches the ground truth string."
# }

# Example 4:
# Passage: Jane was an American magazine created to appeal to the women who grew up reading \"Sassy Magazine\"; Jane Pratt was the founding editor of each.First for Women is a woman's magazine published by Bauer Media Group in the USA.
# Question: Are Jane and First for Women both women's magazines?
# {
#   "restatement": "Answer the binary question strictly with a lowercase 'yes' or 'no'.",
#   "reasoning step": "Step 1: Verify in the passage that both Jane and First for Women are described as women's magazines. Step 2: If both are affirmed, return 'yes'; if not, return 'no'. Step 3: Do not add explanations or punctuation. Step 4: Ensure the output is exactly one token.",
#   "answer": "yes",
#   "self_check": "Conforms to lowercase 'yes/no' requirement; no extra words or punctuation; aligns with ground truth that both are women's magazines."
# }
# """
halueval_6_FEW_SHOT = """
Example 1:  # 精确实体抽取（别名 vs 全名）
Passage: Vocals are handled by Aesop Rock, with guest appearances from Camu Tao and Metro of S.A. Smash and Definitive Jux label head El-P. Jaime Meline (born March 2, 1975), better known by his stage name El-P, is an American hip hop recording artist, record producer, and record executive.
Question: Fast Cars, Danger, Fire and Knives includes guest appearances from which hip hop record executive?
{
  "restatement": "Identify and return the exact full name of the hip hop record executive mentioned.",
  "reasoning step": "Step 1: Find the sentence listing guest appearances. Step 2: Extract the canonical full name, not the alias. Step 3: Output only the name string with correct casing and spacing.",
  "answer": "Jaime Meline",
  "self_check": "Matches the passage exactly; no alias, abbreviation, or extra text."
}

Example 2:  # 数字/单位规范化
Passage: The 2013 Liqui Moly Bathurst 12 Hour race was staged on a 6.213 km circuit at Mount Panorama.
Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
{
  "restatement": "Return the length of the track with proper unit and format.",
  "reasoning step": "Step 1: Identify the numeric value of track length. Step 2: Ensure the answer includes 'km long'. Step 3: Output exactly as in passage format.",
  "answer": "6.213 km long",
  "self_check": "Includes both numeric value and unit exactly; no extra words."
}

Example 3:  # 列表/多实体答案
Passage: The band consists of James Hetfield (vocals/guitar), Lars Ulrich (drums), Kirk Hammett (lead guitar), and Robert Trujillo (bass).
Question: What are the names of the current members of the American heavy metal band?
{
  "restatement": "List all current members of the band in passage order.",
  "reasoning step": "Step 1: Identify each member's full name. Step 2: Preserve the order as in passage. Step 3: Separate names with commas only; do not add extra descriptions.",
  "answer": "James Hetfield, Lars Ulrich, Kirk Hammett, Robert Trujillo",
  "self_check": "All names included; order preserved; no alias or extra words."
}

Example 4:  # 单词/短语输出
Passage: Nicholas Ray and Elia Kazan were both influential filmmakers in Hollywood.
Question: What profession does Nicholas Ray and Elia Kazan have in common?
{
  "restatement": "Return the common profession as a single word or short phrase.",
  "reasoning step": "Step 1: Identify the profession of both individuals. Step 2: Output only the profession word, no sentence.",
  "answer": "director",
  "self_check": "Single-word answer; no sentence or extra text."
}

Example 5:  # 日期/先后判断
Passage: Sir Francis Nethersole was born in 1587. Elizabeth Stuart was born in 1596.
Question: Who was born first, Francis Nethersole or Elizabeth Stuart?
{
  "restatement": "Determine which person was born earlier.",
  "reasoning step": "Step 1: Extract birth years. Step 2: Compare the years. Step 3: Output the full name of the earlier-born individual.",
  "answer": "Sir Francis Nethersole",
  "self_check": "Uses full name as in passage; no extra text."
}

Example 6:  # 格式/标准化注意（避免多余描述）
Passage: Chang Ucchin was born in Korea during a period that ended with World War II.
Question: During which historical event did Chang Ucchin's early life period end?
{
  "restatement": "Return the historical event name exactly as in passage.",
  "reasoning step": "Step 1: Identify the event mentioned as ending the period. Step 2: Output exactly the event name without additional year or description.",
  "answer": "World War II",
  "self_check": "Matches passage exactly; no extra explanation or year added."
}
"""


covidQA_4_FEW_SHOT = """
Characteristics of the covidQA dataset:
- Passages are excerpts from biomedical/public health papers.
- Questions require direct fact extraction: exact phrases, full entity names, complete sentences, or lists.
- Rules: (1) Use EXACT WORDING from the passage. (2) For entities, use the FULL NAME (e.g., "Human metapneumovirus (HMPV)", not just "HMPV"). (3) For Yes/No questions, extract the specific passage text answering the question. (4) For lists, include ALL items. (5) Do NOT paraphrase.
- Your FINAL line must be: Answer:[copied text from passage]

Example 1: (Entity with full name — do NOT abbreviate)
Passage: Abstract: Human metapneumovirus (HMPV) is a major cause of acute respiratory illness in young children worldwide.

Text: HMPV causes a spectrum of respiratory disease ranging from mild upper respiratory infection to bronchiolitis and pneumonia.

Question: What causes acute respiratory illness in young children?
The passage names the cause in the abstract with the full name.
Answer: Human metapneumovirus (HMPV)

Example 2: (Yes/No question — extract specific passage text, NOT just "Yes" or "No")
Passage: Abstract: The 2020 Diamond Princess cruise ship outbreak.

Text: The movement restriction policy from 5 February 2020 was highly successful in greatly reducing the number of secondary transmissions on board the Diamond Princess cruise ship.

Question: What was the effect of movement restriction policy on the Diamond Princess?
The question asks for effect. Copy the relevant sentence from the passage.
Answer: highly successful in greatly reducing the number of secondary transmissions on board

Example 3: (List — include ALL items in the list)
Passage: Abstract: The 2014 Ebola epidemic.

Text: The epidemic began in Guinea in December 2013 and spread rapidly to Sierra Leone and Liberia. Within months, cases were reported in neighbouring sub-Saharan countries, North America, and Europe, sparking international concern.

Question: Where did the 2014 Ebola epidemic in West Africa spread to?
The passage lists ALL the destinations. Copy them all.
Answer: neighbouring sub-Saharan countries, North America, and Europe

Example 4: (Find the SPECIFIC mention, not other examples in the passage)
Passage: Abstract: RNA-binding proteins (RBPs) regulate inflammatory responses.

Text: RBPs include members of the Roquin and Regnase families, which post-transcriptionally control expression of crucial inflammatory factors. Other RBPs include tristetraprolin and AUF1, which promote mRNA decay.

Question: What do RBPs include?
The passage first defines RBPs by the Roquin/Regnase families. This is the primary answer.
Answer: members of the Roquin and Regnase families
"""


financebench_5_FEW_SHOT = """
Characteristics of the financebench dataset:
- Focuses on real-world financial statements, earnings reports, and segment disclosures from public companies, requiring precise extraction, calculation, and reasoning strictly based on the provided tables or passages.
- Questions often involve multi-step numerical reasoning, ratio or margin calculation, period comparison, or entity identification, with answers in the form of numbers, percentages, entity names, or short factual statements.
- All answers must be strictly based on the passage or table, with no guessing, extrapolation, or use of external knowledge.
- If the passage does not provide enough information, the model should refuse to answer and not make assumptions.

Example 1:
Passage: "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA... (full PG&E income statement for FY2020-2022)"
Question: "What is the FY2020 - FY2022 3 year average of COGS as a % of revenue for PG&E? Answer in units of percents and round to one decimal place."
Step-by-step reasoning:
1. Extract COGS and total revenue for each year from the income statement.
2. Compute annual COGS %: COGS / Revenue * 100 for each year.
3. Calculate the mean of the three percentages.
4. Round the result to one decimal place, using only provided data.
Final Answer: 21.6%

Example 2:
Passage: "Segment results managed basis... (full JPM segment results table)"
Question: "Which of the business segments of JPM had the highest net income for the nine months ended September 30, 2022?"
Step-by-step reasoning:
1. Extract net income values for each business segment from the table.
2. Compare net income across all segments.
3. Identify the segment with the highest net income.
4. Use only the provided table data.
Final Answer: Corporate & Investment Bank. Its net income was $11,642 million.

Example 3:
Passage: "ORACLE CORPORATION CONSOLIDATED BALANCE SHEETS... (full FY2021 and FY2020)"
Question: "What is Oracle's year end FY2021 total current assets (in USD billions)?"
Step-by-step reasoning:
1. Locate the balance sheet section for May 31, 2021.
2. Find the value listed for total current assets.
3. Convert to USD billions if necessary.
4. Report the value directly from the balance sheet.
Final Answer: $55.57

Example 4:
Passage: "3M Company Consolidated Statement of Income... (full FY2020-2022 income statement)"
Question: "What is the FY2022 unadjusted operating income % margin for 3M?"
Step-by-step reasoning:
1. Locate operating income and net sales for FY2022.
2. Compute operating income % margin = Operating Income / Net Sales * 100.
3. Round the result to one decimal place.
4. Ensure calculation uses only the data in the income statement.
Final Answer: 19.1%

Example 5:
Passage: "Best Buy Consolidated Statements of Earnings & Cash Flows... (full FY2020-2022)"
Question: "What is the FY2020 - FY2022 3 year average unadjusted EBITDA % margin for Best Buy? Define unadjusted EBITDA as unadjusted operating income + depreciation and amortization from the cash flow statement."
Step-by-step reasoning:
1. For each year, add operating income (P&L) and depreciation & amortization (cash flow statement) to get unadjusted EBITDA.
2. Divide each year's unadjusted EBITDA by revenue to compute annual margin.
3. Compute the 3-year average of these margins.
4. Round the result to one decimal place, using only provided statements.
Final Answer: 6.9%
"""


pubmedQA_4_FEW_SHOT = """
Example 1:
Passage: Ultrasound is currently not established for the diagnosis of fractures. The aim of this study was to compare ultrasound and X-ray beyond their use solely for the identification of fractures, i.e., for the detection of fracture type and dislocation for pediatric fracture diagnosis. Limb bones of dead young pigs served as a model for pediatric bones. The fractured bones were examined with ultrasound, X-ray, and CT, which served as the gold standard. 162 of 248 bones were fractured. 130 fractures were identified using ultrasound, and 148 using X-ray. There were some advantages of X-ray over ultrasound in the detection of fracture type (80 correct results using X-ray, 66 correct results using ultrasound). Ultrasound, however, was superior to X-ray for dislocation identification (41 correct results using X-ray, 51 correct results using ultrasound). Both findings were not statistically significant after adjustment for multiple testing.
Question: Is ultrasound equal to X-ray in pediatric fracture diagnosis?
Answer: Yes. Ultrasound and X-ray are comparable for fracture identification, type, and dislocation in pediatric fractures.

Example 2:
Passage: Although desflurane is commonly used to control surgically induced hypertension, its effects on left ventricular (LV) function have not been investigated in this clinical situation. The purpose of the present study was to evaluate the LV function response to desflurane, when used to control intraoperative hypertension. In 50 patients scheduled for vascular surgery, anesthesia was induced with sufentanil 0.5 microg/kg, midazolam 0.3 mg/kg and atracurium 0.5 mg/kg. After tracheal intubation, anesthesia was maintained with increments of drugs with controlled ventilation (N2O/O2=60/40%) until the start of surgery. A 5 Mhz transesophageal echocardiography (TEE) probe was inserted after intubation. Pulmonary artery catheter and TEE measurements were obtained after induction (control value), at surgical incision (t1) if it was associated with an increase in systolic arterial pressure (SAP) greater than 140 mmHg (hypertension) and after control of hemodynamic parameters by administration of desflurane (return of SAP to within 20% of control) in a fresh gas flow of 3 l/min. Sixteen patients developed hypertension at surgical incision. SAP was controlled by desflurane in all 16 patients. Afterload assessed by systemic vascular resistance index (SVRI), end-systolic wall-stress (ESWS) and left-ventricular stroke work index (LVSWI) increased with incision until hypertension returned to post-induction values with mean end-tidal concentration of 5.1+/-0.7% desflurane. No change in heart rate, cardiac index, mean pulmonary arterial pressure, stroke volume, end-diastolic and end-systolic cross-sectional areas, fractional area change and left ventricular circumferential fiber shortening was noted when desflurane was added to restore blood pressure.
Question: Does desflurane alter left ventricular function when used to control surgical stimulation during aortic surgery?
Answer: No. Desflurane does not significantly alter left ventricular function during surgical hypertension control.

Example 3:
Passage: The differential diagnosis between essential tremor (ET) and Parkinson's disease (PD) may be difficult on clinical grounds alone. A small percentage of ET patients presenting symptoms of possible PD may progress to typical parkinsonism. FP-CIT (DaTSCAN) has been proven useful in differentiating PD from ET. 28 ET patients and 28 normal volunteers were studied. 33 PD patients with unilateral limb involvement were also included. SPECT data showed no difference between NV and ET patients. Reduced accumulation was found in P of all PD patients. Semiquantitative analysis revealed significant differences between NV and ET patients in the striatum, reduced in ET. Follow-up in ET patients showed no significant change.
Question: SPECT study with I-123-Ioflupane (DaTSCAN) in patients with essential tremor. Is there any correlation with Parkinson's disease?
Answer: No. ET patients do not show SPECT patterns indicative of Parkinson's disease correlation.

Example 4:
Passage: To determine if composite measures based on process indicators are consistent with short-term outcome indicators in surgical colorectal cancer care. Longitudinal analysis for 85 Dutch hospitals included 4732 colon and 2239 rectum carcinoma patients. All available process indicators were aggregated into five composite measures. Associations with risk-adjusted postoperative mortality and morbidity were analyzed at patient and hospital level. At patient level, only one measure was negatively associated with morbidity for rectum carcinoma. At hospital level, strong negative associations were found between composite measures and hospital mortality/morbidity rates.
Question: Are composite process indicators consistent with short-term outcomes in surgical colorectal cancer care?
Answer: Maybe. Patient-level association is inconsistent, but hospital-level composite measures align with improved short-term outcomes.
"""


RAGTruth_5_FEW_SHOT = """
Characteristics of the RAGTruth dataset:
- Focuses on open-domain, multi-passage question answering, where answers must be strictly supported by the given passages.
- Passages may contain redundant, partial, or tangential information; the model must identify and synthesize only directly relevant evidence.
- Questions can be factual, procedural, or entity-based; answers must not rely on any external knowledge or inference.
- If the answer cannot be determined from the passages, respond: "Unable to answer based on given passages."

Example 1: (Definition)
Passage 1: Deoxyribonucleic acid (DNA) is a nucleic acid that contains genetic instructions for all known living organisms and some viruses. Its main role is the long-term storage of information.
Passage 2: DNA is composed of nucleotides, each containing a phosphate group, a sugar group, and a nitrogen base. The four nitrogen bases are adenine (A), thymine (T), guanine (G), and cytosine (C). The sequence of these bases determines DNA's genetic code.
Passage 3: DNA has a double helix structure formed by base pairs attached to a sugar-phosphate backbone and can replicate itself.

Question: What is DNA?
{
  "restatement": "This question asks for a definition of DNA strictly based on the given passages.",
  "reasoning_step": "Step 1: Identify all relevant statements describing DNA from the passages. Step 2: Synthesize the description without adding any outside knowledge. Step 3: Ensure all details (structure, function, composition) are included.",
  "answer": "DNA, or deoxyribonucleic acid, is a nucleic acid containing genetic instructions for all known living organisms and some viruses. It is composed of nucleotides, each consisting of a phosphate group, a sugar group, and a nitrogen base (adenine, thymine, guanine, or cytosine). The sequence of these bases determines the genetic code. DNA has a double helix structure with base pairs attached to a sugar-phosphate backbone and can replicate itself.",
  "self_check": "Answer is fully supported by the passages and contains all key details; no external information is added."
}

Example 2: (Procedure)
Passage 1: To fold a quilt, lay the quilt on a flat surface. Fold the lower right corner up towards the center on the bias. Fold the lower left corner up towards the center on the bias.
Passage 2: Fold the top left corner down towards the center on the bias. Fold the top right corner down towards the center on the bias.

Question: How to fold a quilt?
{
  "restatement": "This question asks for the step-by-step folding instructions based strictly on the passages.",
  "reasoning_step": "Step 1: Extract each folding instruction in the order given. Step 2: Make sure all steps are included and clearly presented.",
  "answer": "1. Lay the quilt on a flat surface. 2. Fold the lower right corner up towards the center on the bias. 3. Fold the lower left corner up towards the center on the bias. 4. Fold the top left corner down towards the center on the bias. 5. Fold the top right corner down towards the center on the bias.",
  "self_check": "All folding steps are directly taken from the passages, in correct order."
}

Example 3: (Refusal)
Passage 1: Dump a chimney of lit coals into the grill. Spread coals into a double layer at the far side and single layer in the center. Leave the section closest to you coal-free. Replace the grill grate.
Passage 2: If using a gas grill, set it up for three-zone grilling. Keep the grate hot, clean, and lubricated.
Passage 3: The Porterhouse is two steaks in one, with a NY Strip on one side and filet on the other. The bone provides flavor.

Question: How to grill a porterhouse?
{
  "restatement": "This question asks for instructions to grill a Porterhouse steak based strictly on passages.",
  "reasoning_step": "Step 1: Search all passages for direct instructions on grilling a Porterhouse. Step 2: Do not infer or add information not present in passages.",
  "answer": "Unable to answer based on given passages.",
  "self_check": "Passages do not contain specific instructions for grilling a Porterhouse, so the answer correctly refuses to answer."
}

Example 4: (Comparison / Numerical)
Passage 1: The Eiffel Tower is 324 meters tall. The Statue of Liberty is 93 meters tall, including its pedestal.
Passage 2: Both landmarks are popular tourist attractions in their respective countries.

Question: Which landmark is taller, the Eiffel Tower or the Statue of Liberty?
{
  "restatement": "This question asks to compare heights strictly based on the passages.",
  "reasoning_step": "Step 1: Identify the height of each landmark from the passages. Step 2: Compare the numeric values. Step 3: Determine which is taller.",
  "answer": "The Eiffel Tower is taller than the Statue of Liberty.",
  "self_check": "Comparison is based strictly on passage data: 324m > 93m. No external knowledge used."
}

Example 5: (Counting / Numerical)
Passage 1: The zoo has lions, tigers, and bears. There are 4 lions, 3 tigers, and 5 bears.
Passage 2: The zoo recently acquired 2 more lions and 1 more tiger.

Question: How many lions are now in the zoo?
{
  "restatement": "This question asks for the updated number of lions strictly based on the passages.",
  "reasoning_step": "Step 1: Extract the initial number of lions. Step 2: Identify any additions. Step 3: Compute the total number of lions.",
  "answer": "There are 6 lions in the zoo.",
  "self_check": "Calculation uses only passage data: 4 initial + 2 added = 6. No external assumptions."
}
"""
