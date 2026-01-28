HISTORY_8_FEW_SHOT =  """
Characteristics of the DROP history dataset:

CRITICAL GUIDELINES:
1. Question Analysis: Carefully examine the question to determine the expected answer type:
   - 'how many' → number (count)
   - 'what year' → specific year from passage
   - 'what', 'which', 'who' → name, phrase, or span from passage
   - 'when' → time period or date from passage
2. Number Handling: Only use numbers that explicitly appear in the passage.
   - Do not assume or invent numbers for calculation
   - Pay attention to time periods and organizational changes
   - For time calculations, use exact years mentioned
3. Information Extraction: Extract precise information from the passage.
   - Names: Use exact names as they appear
   - Numbers: Use exact numbers as stated
   - Events: Use specific events mentioned
4. Insufficient Information: If the passage does not provide enough information to answer, respond with:
   'The passage does not provide enough information to answer this question.'
5. Multiple Choice: If the question contains 'or', pay close attention and answer using the options mentioned in the question itself.

Example 1:
Passage: John II of France was captured in 1356 during the Battle of Poitiers. The Navarrese rebellion erupted in 1365, shortly after his captivity ended.
Question: How many years before Navarrase used the captivity of John II?
{
  "restatement": "The question asks for the number of years between John II's captivity and the Navarrese rebellion.",
  "reasoning step": "Step 1: Identify the year of John II's captivity (1356). Step 2: Identify the year of the Navarrese rebellion (1365). Step 3: Calculate the difference: 1365 - 1356 = 9 years. Step 4: Ensure the calculation is based solely on the passage.",
  "answer": "9",
  "self_check": "The question asks for a number; the answer '9' is numeric and based on the passage."
}

Example 2:
Passage: During the Mongol invasions of Goryeo, the king fled to Ganghwa Island to escape the advancing forces. The Mongol attacks occurred intermittently before and after the king's flight.
Question: What happened first: intermittent invasion of Goryeo or fleeing of king to Ganghwa Island?
{
  "restatement": "The question asks which event occurred first: the intermittent invasions or the king's flight to Ganghwa Island.",
  "reasoning step": "Step 1: Identify the timeline of the Mongol invasions. Step 2: Identify when the king fled. Step 3: Compare the two events based on the passage. Step 4: Determine which came first.",
  "answer": "intermittent invasion of Goryeo",
  "self_check": "The answer is an event name, correctly reflecting the passage information and question type."
}

Example 3:
Passage: The United States faced a shortage of voluntary enlistment during the early years of the war, which led to Congress implementing conscription to meet military needs.
Question: What was inadequate?
{
  "restatement": "The question asks what was insufficient or inadequate based on the passage.",
  "reasoning step": "Step 1: Identify what was described as insufficient. Step 2: The passage states that voluntary enlistment was not enough. Step 3: Confirm that no other factors are mentioned as inadequate. Step 4: Ensure answer strictly matches the passage.",
  "answer": "voluntary enlistment",
  "self_check": "The question asks for a reason/quantity inadequacy; the answer matches the passage."
}

Example 4:
Passage: After the Treaty of Fez in 1912, several regions of Morocco continued to resist French control. These revolts persisted for more than two decades until 1934.
Question: How many years did the regions remain in revolt after the Treaty of Fez?
{
  "restatement": "The question asks for the duration of resistance in Morocco after the Treaty of Fez.",
  "reasoning step": "Step 1: Identify the year of the Treaty of Fez (1912). Step 2: Identify the year when revolts ended (1934). Step 3: Calculate the duration: 1934 - 1912 = 22 years. Step 4: Ensure calculation is based solely on passage information.",
  "answer": "22",
  "self_check": "The question asks for a number; the answer '22' matches the passage information."
}

Example 5:
Passage: The Lithuanian Civil War began in 1381 and concluded in 1387, marking six years of conflict between competing factions.
Question: How many years did the Lithuanian Civil War last?
{
  "restatement": "The question asks for the duration of the Lithuanian Civil War based on the passage.",
  "reasoning step": "Step 1: Identify the start year of the civil war (1381). Step 2: Identify the end year (1387). Step 3: Calculate the duration: 1387 - 1381 = 6 years. Step 4: Ensure the answer is based only on passage information.",
  "answer": "6",
  "self_check": "The question asks for a number; the answer '6' is numeric and matches the passage."
}
"""



NFL_8_FEW_SHOT = """
CRITICAL GUIDELINES:
1. Question Analysis: Carefully examine the question to determine the expected answer type:
   - 'how many' → number (count)
   - 'what year' → specific year from passage
   - 'what', 'which', 'who' → name, phrase, or span from passage
   - 'when' → time period or date from passage

2. Number Handling: Only use numbers that explicitly appear in the passage.
   - Do not assume or invent numbers for calculation
   - Pay attention to time periods and organizational changes
   - For time calculations, use exact years mentioned

3. Information Extraction: Extract precise information from the passage.
   - Names: Use exact names as they appear
   - Numbers: Use exact numbers as stated
   - Events: Use specific events mentioned

4. Insufficient Information: If the passage does not provide enough information to answer, respond with:
   'The passage does not provide enough information to answer this question.'

5. Multiple Choice: If the question contains 'or', pay close attention and answer using the options mentioned in the question itself.

Example 1:
Passage: Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.
Question: How many games had the Bills won before this game?
{
  "restatement": "Determine the number of games the Bills had won before this Week 3 game.",
  "reasoning step": "The passage says 'Trying to snap a two-game skid,' which means the Bills lost the previous 2 games. If they are trying to snap a losing streak, that implies they have not won any games in this season so far.",
  "answer": "0"
}


Example 2:
Passage: Hoping to rebound from their tough overtime road loss to the Raiders, the Jets went home for a Week 8 duel with the Kansas City Chiefs. In the first quarter, New York took flight as QB Brett Favre completed an 18-yard TD pass to RB Leon Washington. In the second quarter, the Chiefs tied the game as QB Tyler Thigpen completed a 19-yard TD pass to TE Tony Gonzalez. The Jets would ground_truth with Washington getting a 60-yard TD run. Kansas City closed out the half as Thigpen completed an 11-yard TD pass to WR Mark Bradley.
Question: How many points did each team have at halftime?
{
  "restatement": "Find the halftime points for both the Jets and the Chiefs.",
  "reasoning step": "Jets scored 7 in Q1 (TD + extra point), 7 in Q2 (TD + extra point), totaling 14. Chiefs scored 7 in Q2 (TD + extra point), then another 7 in Q2, totaling 14.",
  "answer": "Jets 14, Chiefs 14"
}

Example 3:
Passage: Coming off their road win over the Redskins, the Chiefs went home, donned their Dallas Texans throwbacks, and played a Week 7 AFL Legacy game with the San Diego Chargers. Kansas City would find themselves trailing in the first quarter as Chargers quarterback Philip Rivers completed a 3-yard touchdown pass to wide receiver Malcom Floyd, followed by a 10-yard touchdown pass to wide receiver Vincent Jackson.
Question: How short was the shortest touchdown pass?
{
  "restatement": "Identify the shortest touchdown pass in the game.",
  "reasoning step": "The passage mentions two touchdown passes: one of 3 yards to Malcom Floyd, another of 10 yards to Vincent Jackson. The shortest is 3 yards.",
  "answer": "3-yard touchdown pass to wide receiver Malcom Floyd"
}


Example 4:
Passage: Coming off their road win over the Lions, the Steelers went home for a divisional match with the Cleveland Browns. After a scoreless first quarter, Pittsburgh came out striking in the second quarter as quarterback Ben Roethlisberger completed an 8-yard touchdown pass to tight end Heath Miller and a 52-yard touchdown pass to wide receiver Hines Ward.
Question: How many is the difference in the yards of the TD pass to Miller and the yards of the TD pass to Ward?
{
  "restatement": "Calculate the difference in yardage between two touchdown passes.",
  "reasoning step": "The TD pass to Miller was 8 yards. The TD pass to Ward was 52 yards. Difference = 52 - 8 = 44 yards.",
  "answer": "44"
}


Example 5:
Passage: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal. In the fourth quarter, the Panthers sealed the win in the final seconds with Kasay nailing a 52-yard field goal as time ran out.
Question: How many yards was the longest field goal of the game?
{
  "restatement": "Determine the distance of the longest field goal in the game.",
  "reasoning step": "The field goals listed are 23, 25, 35, and 52 yards. The longest is 52 yards.",
  "answer": "52-yard"
}

"""

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
- Answers must be short exact spans from the passage (word/phrase/number).
- Do NOT paraphrase, explain, or add context.
- If multiple items are listed, include only those mentioned—no additions.
- If the passage does not contain enough information, answer: "Unable to answer based on given information."
- Always end with: Answer:[your answer here].

Example 1:
Question: What is interleukin-1Beta?
Passage: Interleukin-1Beta is described in the study as pro-inflammatory cytokines.
Answer: Answer: pro-inflammatory cytokines

Example 2:
Question: What do RBPs include?
Passage: RBPs include members of the Roquin and Regnase families.
Answer: Answer: members of the Roquin and Regnase families

Example 3:
Question: In this study, how did treatment of EAP after infection affect survival?
Passage: Treatment of EAP after infection did not provide a survival advantage.
Answer: Answer: did not provide a survival advantage

Example 4:
Question: What family of virus does SARS reside in?
Passage: SARS resides in the coronavirus family.
Answer: Answer: coronavirus

Example 5:
Question: How to grill a porterhouse?
Passage: The passages mention coals, grill setup, and Porterhouse definition, but do not provide specific instructions for grilling a Porterhouse.
Answer: Answer: Unable to answer based on given information.
"""



financebench_5_FEW_SHOT = """
Example 1:
Passage: Boeing consolidated income statement excerpt: Interest expense (FY2017) reported as $360,000 (USD thousands).
Question: How much was Boeing's FY2017 total interest expense (in USD thousands)?
Answer: $360000.00

Example 2:
Passage: Balance sheet excerpt: Total current assets = $9,000; Total current liabilities = $8,910.
Question: What is FY2019 working capital ratio for Costco? Define as current assets / current liabilities.
Answer: 1.01

Example 3:
Passage: 3M consolidated income statement (FY2022): Net sales = $34,229; Operating income = $6,539.
Question: What is 3M's FY2022 unadjusted operating income % margin?
Answer: 19.1%

Example 4:
Passage: PG&E income statement excerpt (2020-2022) — COGS and total revenues available for each year.
Question: What is the FY2020 - FY2022 3 year average of COGS as a % of revenue for PG&E? Round to 1 decimal.
Answer: 21.6%

Example 5:
Passage: Best Buy P&L and Cash Flow (FY2020-FY2022): operating income and D&A for each year and revenues present.
Question: What is the FY2020 - FY2022 3 year average unadjusted EBITDA % margin for Best Buy? Round to 1 decimal.
Answer: 6.9%
"""


pubmedQA_4_FEW_SHOT = """
Example 1: 
Passage: We investigated the role of surgical ablation targeting the autonomous nervous system during a Cox-Maze IV procedure in the maintenance of sinus rhythm at long-term follow-up. The patient population consisted of 519 subjects with persistent or long-standing persistent atrial fibrillation (AF) undergoing radiofrequency Maze IV during open heart surgery between January 2006 and July 2013 at three institutions without (Group 1) or with (Group 2) ganglionated plexi (GP) ablation. Recurrence of atrial fibrillation off-antiarrhythmic drugs was the primary outcome. Predictors of AF recurrence were evaluated by means of competing risk regression. Median follow-up was 36.7 months. The percentage of patients in normal sinus rhythm (NSR) off-antiarrhythmic drugs did not differ between groups (Group 1-75.5%, Group 2-67.8%, p = 0.08). Duration of AF ≥ 38 months (p = 0.01), left atrial diameter ≥ 54 mm (0.001), left atrial area ≥ 33 cm(2) (p = 0.005), absence of connecting lesions (p= 0.04), and absence of right atrial ablation (p<0.001) were independently associated with high incidence of AF recurrence. In contrast the absence of GP ablation was not a significant factor (p = 0.12).
Question: Is ganglionated plexi ablation during Maze IV procedure beneficial for postoperative long-term stable sinus rhythm?
{
  "restatement": "Determine if GP ablation improves long-term NSR after Maze IV.",
  "reasoning step": "Step 1: Compare NSR rates between groups with and without GP ablation. Step 2: Note statistical insignificance (p=0.12). Step 3: Conclude based on study findings.",
  "answer": "No. GP ablation did not prove to be beneficial for postoperative stable NSR. A complete left atrial lesion set and biatrial ablation are advisable for improving rhythm outcomes. Randomized controlled trials are necessary to confirm our findings.",
  "self_check": "Starts with 'No.'; conclusion ≤25 words; uses terms from passage; no extra info"
}

Example 2:
Passage: Diabetes mellitus (DM) is undiagnosed in approximately half of the patients actually suffering from the disease. In addition, the prevalence of DM is more than twice as high as in patients with periodontitis when compared to periodontally healthy subjects. Thus, a high number of patients with periodontitis may have undiagnosed DM. The purpose of the present study was to evaluate whether blood oozing from a gingival crevice during routine periodontal examination can be used for determining glucose levels. Observational cross-sectional studies were carried out in 75 patients (43 males and 32 females) with chronic periodontitis who were divided into two groups: Group I and Group II, respectively. Blood oozing from the gingival crevices of anterior teeth following periodontal probing was collected with the stick of glucose self-monitoring device, and the blood glucose levels were measured. At the same time, finger-prick blood was taken for glucometric analysis and subsequent readings were recorded. The patient's blood glucose values ranged from 74 to 256 mg/dl. The comparison between gingival crevicular blood and finger-prick blood showed a very strong correlation, with a t value of 3.97 (at P value = 0.001).
Question: Can gingival crevicular blood be relied upon for assessment of blood glucose level?
{
  "restatement": "Determine if GCB can be used to measure blood glucose reliably.",
  "reasoning step": "Step 1: Identify that GCB was collected and compared to finger-prick blood. Step 2: Note strong correlation and statistical significance. Step 3: Conclude based on study findings.",
  "answer": "Yes. The data from this study has shown that GCB collected during diagnostic periodontal examination can be an excellent source of blood for glucometric analysis.",
  "self_check": "Starts with 'Yes.'; conclusion ≤25 words; uses terms from passage; no extra info"
}

Example 3:
Passage: Optimization of the preoperative hemoglobin (Hb) level is an effective way to reduce allogeneic transfusion in total knee arthroplasty (TKA) though the procedure is expensive, requires close monitoring and is often inconvenient for patients with reduced mobility. All consecutive patients who undergone primary TKA in our center over 2 years, and received tranexamic acid intraoperatively, were reviewed. The adjusted association between preoperative Hb levels and transfusion was assessed by multivariate logistic regression, and the estimated probability of transfusion for individual patients was derived from the logistic model. Out of the 784 patients who meet the inclusion criteria, risk of transfusion was associated with poorer performance status, as measured by the America Association of Anestesiology (ASA) score III/IV (OR: 3·3, P < 0·001) and lower preoperative Hb level (OR 3·8 for each g/dl below 13 g/dl; P < 0·001). According to the Hb level, the estimated probability of transfusion was 0·03 (range: 0·03-0·64) for ASA I/II patients and 0·10 (range: 0·10-0·84) for ASA III/IV.
Question: Should all patients be optimized to the same preoperative hemoglobin level to avoid transfusion in primary knee arthroplasty?
{
  "restatement": "Decide if uniform preoperative Hb targets are necessary for all TKA patients.",
  "reasoning step": "Step 1: Analyze association between Hb levels, ASA score, and transfusion risk. Step 2: Note variation in transfusion probability across patients. Step 3: Conclude based on study findings.",
  "answer": "No. Not all the patients undergoing TKA who receive tranexamic acid need the same preoperative Hb optimization target. Two easily available factors, such as the ASA score and the Hb level, can help individualize the Hb optimization target.",
  "self_check": "Starts with 'No.'; conclusion ≤25 words; uses terms from passage; no extra info"
}

Example 4:
Passage: To assess the relationship between the experience of pediatric housestaff and tests ordered on infants in the neonatal intensive care unit (ICU). Prospective, cohort study over one full academic year. One academic Level III neonatal intensive care nursery. Data were collected prospectively on all 785 infants admitted to the neonatal ICU from July 1993 to June 1994. These infants were cared for by 14 different categorical pediatric housestaff. Our neonatal ICU has either a resident or an intern on-call by himself/herself at night, affording us a natural setting to compare intern vs. resident test ordering. The outcomes of interest were number of arterial blood gases, radiographs, and electrolytes ordered per infant by the on-call pediatric houseofficer, as tabulated the morning after the call night. Control variables included the severity-of-illness of the individual infant (using the Neonatal Therapeutic Intervention Scoring System), the workload of the houseofficer (number of patients, number of admissions), and supervision (rounding frequency and on-call attending). Controlling for the severity-of-illness of the infant, the workload on the call night, and supervision with multiple linear regression, we found that interns ordered significantly (p = .02) greater numbers of arterial blood gases per infant than residents, amounting to some 0.33 blood gases per infant per call night (3.22 vs. 2.89 arterial blood gases per infant per night). This increase of 0.33 blood gases per infant amounts to interns ordering $169 more arterial blood gases per call night at our institution. There was no difference between interns and residents in ordering radiographs or electrolytes.
Question: Does pediatric housestaff experience influence tests ordered for infants in the neonatal intensive care unit?
{
  "restatement": "Determine whether housestaff experience affects test ordering in NICU infants.",
  "reasoning step": "Step 1: Compare test orders between interns and residents. Step 2: Identify significant differences for arterial blood gases. Step 3: Conclude based on study findings.",
  "answer": "Yes. Interns order significantly more arterial blood gases per infant than junior and senior residents on-call in the neonatal ICU. Additional study is required to see if the experience of housestaff is associated with a broader array of neonatal outcomes, such as morbidity and mortality.",
  "self_check": "Starts with 'Yes.'; conclusion ≤25 words; uses terms from passage; no extra info"
}

"""


RAGTruth_5_FEW_SHOT = """
Example 1:
Passage 1: DNA is a nucleic acid that contains genetic instructions for living organisms and some viruses.
Passage 2: DNA is composed of nucleotides, each containing a phosphate group, a sugar, and a nitrogen base (A, T, G, C).
Passage 3: DNA forms a double helix and can replicate.
Question: What is DNA?
{
  "reasoning step": Step 1: Identify DNA as a nucleic acid. Step 2: Include composition (nucleotides, phosphate, sugar, bases A/T/G/C). Step 3: Include structure (double helix) and replication ability. Step 4: Combine all into a single concise answer.
  "answer": DNA, or deoxyribonucleic acid, is a nucleic acid storing genetic instructions for organisms and some viruses, composed of nucleotides with phosphate, sugar, and nitrogen bases (A, T, G, C), forming a double helix structure and capable of replication.
}
Example 2:
Passage 1: To fold a quilt, lay it flat on a table or bed.
Passage 2: Fold the lower right corner toward the center along the bias.
Passage 3: Fold the lower left, then top left, then top right corners toward the center, all along the bias.
Question: How to fold a quilt?
{
  "reasoning step": Step 1: Lay quilt flat on a surface. Step 2: Fold corners in order (lower right, lower left, top left, top right) toward the center along the bias. Step 3: Combine steps into a clear instructional answer.
  "answer": To fold a quilt, first lay it flat on a table or bed. Then fold the lower right corner toward the center along the bias, followed by the lower left corner, then the top left corner, and finally the top right corner, each fold following the bias.
}

Example 3:
Passage 1: Automotive technicians can be paid hourly, through commission, or a combination.
Passage 2: Pay varies by specialization, employer, and location.
Passage 3: Some technicians in Alaska earn an average of $23.70 per hour, while those in aerospace products and parts manufacturing may earn $32 per hour.
Question: How do automotive technicians get paid?
{
  "reasoning step": Step 1: Identify payment types (hourly, commission, combination). Step 2: Note variations by specialization, employer, location. Step 3: Include specific examples of pay rates. Step 4: Summarize into complete answer.
  "answer": Automotive technicians are paid using combinations of hourly wages and commission, depending on their specialization, employer, and location. For example, technicians in Alaska earn about $23.70 per hour, while those in aerospace products and parts manufacturing may earn about $32 per hour.
}

Example 4:
Passage: The provided passages do not include any steps or information on grilling a Porterhouse steak.
Question: How to grill a Porterhouse?
{
  "reasoning step": Step 1: Check if passages contain relevant information. Step 2: If no relevant information, provide refusal statement.
  "answer": Unable to answer based on given passages.
}

Example 5:
Passage 1: Magellan of Virginia administers behavioral health services for Medicaid and FAMIS.
Passage 2: It hosts provider Q&A sessions on Fridays from 1:00 to 2:30 PM.
Passage 3: It offers continuing education credits approved by APA and ASWB.
Question: Magellan of Virginia provider
{
  "reasoning step": Step 1: Identify services administered. Step 2: Include Q&A session details. Step 3: Include continuing education credits offered. Step 4: Summarize all into one answer.
  "answer": Magellan of Virginia administers behavioral health services for Medicaid and FAMIS, hosts provider Q&A sessions on Fridays from 1:00 to 2:30 PM, and provides continuing education credits approved by APA and ASWB.
}
"""

