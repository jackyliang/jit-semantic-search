"""Semantic Projection Search (SPS) — the novel JIT search strategy.

Key Idea
--------
Traditional semantic search requires running a neural embedding model on every
document at query time.  That is accurate but slow (~5-50 ms per document).
Traditional lexical search (BM25/TF-IDF) is fast but misses synonyms and
paraphrases.

SPS bridges the gap:

    1. Offline (one-time, cached): learn a LINEAR PROJECTION MATRIX  W  that
       maps cheap sparse text features into approximate neural-embedding space.
       W is learned via least-squares regression on a diverse synthetic corpus:

           W = argmin_W  || X_sparse @ W  -  X_embed ||^2_F

       where X_sparse comes from sklearn's HashingVectorizer (corpus-
       independent — no fitting step!) and X_embed comes from a real
       sentence-embedding model (BAAI/bge-small-en-v1.5 via fastembed).

    2. At query time (JIT):
       a. Compute HashingVectorizer features for query + documents   (< 1 ms)
       b. Multiply by W to project into embedding space              (< 10 ms)
       c. Cosine similarity between projected query and documents     (< 1 ms)

The result is semantic-quality search at near-lexical speed, with NO corpus-
specific pre-processing — perfect for just-in-time scenarios.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

from jit_search.core import JITSearch, SearchResult, SearchStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HashingVectorizer dimensionality.  2**14 = 16 384 features — large enough
# to avoid excessive hash collisions on unigrams+bigrams, small enough that
# the projection matrix W (16384 x embed_dim) fits comfortably in memory.
N_SPARSE_FEATURES = 2**14

# The embedding model used to produce "ground truth" embeddings for training.
# bge-small-en-v1.5 outputs 384-dim vectors and is small/fast to download.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Where the cached projection matrix lives on disk.
CACHE_DIR = Path(os.environ.get("JIT_SEARCH_CACHE_DIR", "~/.cache/jit-search")).expanduser()
MATRIX_CACHE_PATH = CACHE_DIR / "projection_matrix_v3.npz"


# ---------------------------------------------------------------------------
# Training corpus
# ---------------------------------------------------------------------------
# ~2 000 diverse synthetic sentences covering many domains.  The projection
# matrix must generalise across topic, register, and sentence length, so we
# need broad coverage.  Each domain contributes ~100-200 sentences.  These
# are intentionally simple and diverse — the quality of W depends on
# covering the subspaces that real queries and documents will occupy.

def _build_training_corpus() -> list[str]:
    """Return a diverse set of ~2000 synthetic sentences for training W."""

    sentences: list[str] = []

    # ---- Tech support / customer service ----
    sentences.extend([
        "My laptop won't turn on after the latest update",
        "The screen is flickering and showing artifacts",
        "I can't connect to the WiFi network at home",
        "The application crashes every time I open a file",
        "My printer is not responding to print commands",
        "How do I reset my password for the admin portal",
        "The software installation failed with error code 1603",
        "My bluetooth keyboard keeps disconnecting randomly",
        "The hard drive is making clicking noises",
        "I accidentally deleted important files from my desktop",
        "Computer is extremely slow after the system update",
        "Monitor display has dead pixels in the corner",
        "USB ports stopped working on my workstation",
        "Email client keeps asking for credentials repeatedly",
        "VPN connection drops every few minutes during calls",
        "Two-factor authentication is not sending codes",
        "Webcam shows a black screen in video calls",
        "Computer fans are running very loudly at idle",
        "Battery drains in less than two hours",
        "Touchscreen is unresponsive on the tablet device",
        "My account has been locked after failed login attempts",
        "The external hard drive is not being recognized",
        "Audio output has static and crackling sounds",
        "Cannot install the latest security patches",
        "The mouse cursor moves on its own across the screen",
        "System freezes completely when running multiple programs",
        "Unable to access shared network drives from my laptop",
        "The operating system shows blue screen errors frequently",
        "Software license expired and needs renewal",
        "Keyboard shortcuts stopped working after the update",
        "I need help recovering data from a corrupted flash drive",
        "The web browser keeps redirecting to spam sites",
        "Antivirus software detected a trojan on my machine",
        "Print jobs are stuck in the queue and won't clear",
        "The microphone picks up too much background noise",
        "Display brightness controls are not responding",
        "Task manager shows high CPU usage at idle",
        "Cannot sync my calendar across devices",
        "The power adapter makes a buzzing sound when charging",
        "Remote desktop connection keeps timing out",
        "I forgot my encryption key for the secure drive",
        "Application is not compatible with the new OS version",
        "The search function returns no results in the app",
        "Network printer shows offline status despite being powered on",
        "Downloaded file appears corrupted and won't open",
        "Memory usage spikes when opening spreadsheets",
        "The graphic card driver needs to be updated",
        "Can you help me set up email forwarding rules",
        "System clock keeps resetting to the wrong time",
        "File permissions prevent me from editing shared documents",
    ])

    # ---- E-commerce / product descriptions ----
    sentences.extend([
        "Lightweight running shoes with extra arch support",
        "Organic cotton t-shirt available in twelve colors",
        "Noise cancelling headphones with forty hour battery life",
        "Stainless steel water bottle keeps drinks cold for twenty-four hours",
        "Ergonomic office chair with adjustable lumbar support",
        "Wireless charging pad compatible with all modern smartphones",
        "Premium leather wallet with RFID blocking technology",
        "Smart home thermostat that learns your temperature preferences",
        "Bamboo cutting board with juice groove and handle",
        "Ultra-thin laptop sleeve made from recycled materials",
        "Portable bluetooth speaker with waterproof rating",
        "Memory foam pillow designed for side sleepers",
        "Ceramic non-stick frying pan with heat indicator",
        "Adjustable standing desk converter for any workspace",
        "Solar powered portable charger for outdoor adventures",
        "Hand-crafted wooden watch with genuine leather strap",
        "Professional grade chef knife with full tang design",
        "Electric toothbrush with pressure sensor and timer",
        "Compact folding umbrella that fits in your bag",
        "Organic fair trade coffee beans from Colombia",
        "Yoga mat with alignment markers and carrying strap",
        "LED desk lamp with adjustable color temperature",
        "Silicone baking mats that replace parchment paper",
        "Insulated lunch bag with multiple compartments",
        "Glass food storage containers with snap lock lids",
        "Mechanical keyboard with customizable RGB backlighting",
        "Natural deodorant made without aluminum compounds",
        "Cast iron skillet pre-seasoned and ready to use",
        "Smart fitness tracker with heart rate monitoring",
        "Reusable grocery bags made from recycled plastic bottles",
        "Anti-fog swimming goggles with UV protection",
        "Travel adapter compatible with outlets in over 150 countries",
        "Cordless vacuum cleaner with HEPA filtration system",
        "Biodegradable phone case made from plant-based materials",
        "Double-walled espresso cups that keep coffee hot longer",
        "Weighted blanket for improved sleep quality",
        "Digital kitchen scale accurate to one gram",
        "Foldable drone with 4K camera and GPS tracking",
        "High SPF mineral sunscreen safe for coral reefs",
        "Aromatherapy essential oil diffuser with LED lights",
        "Titanium camping cookware set that nests together",
        "Gel ink rollerball pens in assorted colors",
        "Microfiber cleaning cloths that leave no streaks",
        "Collapsible silicone travel cup with leak-proof lid",
        "Plant-based protein powder with no artificial sweeteners",
    ])

    # ---- News / journalism ----
    sentences.extend([
        "Stock markets rallied after the central bank cut interest rates",
        "New climate report warns of accelerating ice sheet melting",
        "Local community raises funds to rebuild historic library",
        "Tech giant announces layoffs affecting ten thousand employees",
        "Scientists discover high-temperature superconductor at ambient pressure",
        "International trade negotiations stall over agricultural tariffs",
        "City council approves new affordable housing development plan",
        "Electric vehicle sales surpass gasoline cars for the first time",
        "Wildfire season expected to be worse due to drought conditions",
        "University researchers develop more efficient solar cell design",
        "Airlines report record passenger numbers during holiday travel",
        "New privacy legislation requires companies to disclose data usage",
        "Hospital system implements artificial intelligence for diagnostics",
        "Space agency confirms plans for crewed Mars mission by 2040",
        "Flooding displaces thousands of residents in the coastal region",
        "Music streaming platform faces antitrust investigation in Europe",
        "Education department announces increased funding for public schools",
        "Professional sports league implements new concussion protocols",
        "Startup raises fifty million dollars in Series B funding",
        "Renewable energy now generates forty percent of electricity supply",
        "New archaeological discovery reveals ancient trade routes",
        "Government proposes carbon tax to reduce greenhouse emissions",
        "Pharmaceutical company recalls medication due to contamination",
        "Self-driving trucks complete first coast-to-coast delivery run",
        "Consumer confidence index drops to its lowest level in a year",
        "Breakthrough gene therapy shows promise for rare childhood disease",
        "Major cyberattack disrupts banking services across the country",
        "Film festival announces diverse lineup of independent movies",
        "National park sets new visitor records during summer months",
        "Labor unions call for higher minimum wage across the sector",
        "Drought conditions threaten crop yields in the midwest",
        "Olympic committee announces new host city for upcoming games",
        "Cryptocurrency regulation gains bipartisan support in congress",
        "Housing prices continue to rise despite higher mortgage rates",
        "Museum opens new wing dedicated to contemporary digital art",
        "Public transit agency unveils plans for light rail expansion",
        "Water treatment plant upgrades to handle increased demand",
        "International summit addresses refugee resettlement challenges",
        "Volcano eruption forces evacuation of nearby island communities",
        "Mobile payment adoption accelerates in developing economies",
    ])

    # ---- Technical documentation / programming ----
    sentences.extend([
        "The function returns a list of dictionaries containing user data",
        "Use the async await pattern for non-blocking IO operations",
        "The database schema includes foreign key constraints for data integrity",
        "Memory leaks can occur if event listeners are not properly removed",
        "Implement pagination to handle large result sets efficiently",
        "The API endpoint accepts JSON payloads with required authentication headers",
        "Use environment variables to store sensitive configuration values",
        "The garbage collector automatically frees unused memory allocations",
        "Apply the singleton pattern when only one instance should exist globally",
        "Hash tables provide constant time lookup for key-value operations",
        "The deployment pipeline runs unit tests before building containers",
        "Binary search requires a sorted array and runs in logarithmic time",
        "Use connection pooling to reduce database connection overhead",
        "The observer pattern decouples event producers from consumers",
        "Buffer overflow vulnerabilities can lead to arbitrary code execution",
        "Type annotations improve code readability and catch errors early",
        "The load balancer distributes incoming requests across server instances",
        "Implement rate limiting to prevent abuse of public API endpoints",
        "Recursive functions must have a proper base case to avoid stack overflow",
        "The cache invalidation strategy determines when stale data gets refreshed",
        "Use mutex locks to prevent race conditions in concurrent programs",
        "The compiler optimizes tail calls to prevent stack frame accumulation",
        "GraphQL allows clients to request exactly the data fields they need",
        "Dependency injection makes code more testable and loosely coupled",
        "The message queue ensures reliable delivery of asynchronous tasks",
        "Use indexes on frequently queried columns to speed up database reads",
        "The virtual machine abstracts hardware differences across platforms",
        "Implement exponential backoff for retrying failed network requests",
        "The parser converts source code tokens into an abstract syntax tree",
        "Use write-ahead logging to ensure database transaction durability",
        "Container orchestration automates deployment scaling and management",
        "The encryption algorithm uses a symmetric key for fast data encoding",
        "Functional programming avoids mutable state and side effects",
        "The proxy pattern controls access to another object transparently",
        "Use feature flags to gradually roll out changes to production users",
        "The TCP handshake establishes a reliable connection between hosts",
        "Microservices communicate through well-defined API contracts",
        "The B-tree index structure enables efficient range queries on disk",
        "Use content delivery networks to serve static assets closer to users",
        "The interpreter reads and executes instructions one at a time",
    ])

    # ---- Medical / health ----
    sentences.extend([
        "The patient reports persistent headaches lasting more than two weeks",
        "Blood pressure readings remain elevated despite medication adjustments",
        "Annual physical examination includes cholesterol screening",
        "The recommended dosage is two tablets twice daily with food",
        "Physical therapy exercises help restore range of motion after surgery",
        "Allergic reactions may include swelling hives and difficulty breathing",
        "Regular cardiovascular exercise reduces the risk of heart disease",
        "The vaccine provides immunity against multiple strains of influenza",
        "MRI results show no abnormalities in the brain tissue",
        "Chronic lower back pain affects quality of life and daily activities",
        "Prenatal vitamins should contain folic acid and iron supplements",
        "The wound requires daily cleaning and fresh bandage changes",
        "Diabetes management includes monitoring blood glucose levels regularly",
        "Symptoms of dehydration include dry mouth fatigue and dizziness",
        "The surgical procedure was completed without complications",
        "Mental health screening is an important part of preventive care",
        "Patients with asthma should avoid exposure to known triggers",
        "The antibiotic course must be completed even if symptoms improve",
        "Sleep apnea diagnosis requires an overnight sleep study evaluation",
        "Nutritional counseling helps patients manage dietary restrictions",
        "Joint inflammation may indicate early stages of arthritis",
        "Immunotherapy treatment shows promising results for certain cancers",
        "Regular dental checkups help prevent cavities and gum disease",
        "The emergency room treated multiple patients after the accident",
        "Thyroid function tests are recommended for unexplained weight changes",
        "Rehabilitation programs support recovery from substance abuse",
        "Skin biopsy results should be available within five business days",
        "Cognitive behavioral therapy is effective for anxiety disorders",
        "The orthopedic surgeon recommends a total knee replacement",
        "Hearing loss assessment includes both audiometry and speech tests",
        "Postoperative care instructions include wound monitoring and rest",
        "Genetic testing can identify predisposition to certain diseases",
        "The pediatrician monitors developmental milestones at each visit",
        "Iron deficiency anemia causes fatigue and shortness of breath",
        "Patients should fast for twelve hours before the blood panel test",
    ])

    # ---- Legal / contracts ----
    sentences.extend([
        "The parties agree to resolve disputes through binding arbitration",
        "This agreement shall be governed by the laws of the state",
        "The tenant must provide thirty days written notice before vacating",
        "Intellectual property rights remain with the original creator",
        "The non-disclosure agreement covers all confidential business information",
        "Force majeure clauses excuse performance during extraordinary events",
        "The warranty period extends for twelve months from date of purchase",
        "Liability is limited to the total amount paid under this contract",
        "The merger requires approval from the board of directors",
        "Employment terminates automatically at the end of the contract period",
        "The licensee may not sublicense rights without written consent",
        "Indemnification covers losses arising from third party claims",
        "The escrow agent holds funds until all conditions are satisfied",
        "Non-compete clauses restrict employment within the same industry",
        "The statute of limitations for filing a claim is three years",
        "Both parties agree to maintain confidentiality of trade secrets",
        "The lease includes an option to renew for an additional term",
        "Severability ensures remaining provisions survive if one is invalid",
        "Due diligence review must be completed before the closing date",
        "The power of attorney authorizes the agent to act on behalf of the principal",
        "Breach of contract entitles the non-breaching party to damages",
        "The trust agreement specifies distribution of assets to beneficiaries",
        "Zoning regulations determine permitted land use in each district",
        "The settlement agreement releases all claims between the parties",
        "Corporate bylaws define the governance structure of the organization",
        "The purchase agreement includes representations and warranties",
        "Mediation is required before initiating formal litigation",
        "The partnership dissolves upon the death or withdrawal of a partner",
        "Consumer protection laws prohibit deceptive advertising practices",
        "The trademark registration grants exclusive rights for specified goods",
    ])

    # ---- Casual conversation / everyday language ----
    sentences.extend([
        "I had the best pizza at that new restaurant downtown",
        "Can you pick up some groceries on your way home",
        "The weather has been amazing this entire week",
        "I just finished watching that documentary everyone recommends",
        "My dog learned a new trick and it's adorable",
        "Let's meet at the coffee shop around three in the afternoon",
        "I've been trying to learn guitar for about six months now",
        "The kids had so much fun at the amusement park yesterday",
        "I think we should repaint the living room this weekend",
        "That concert last night was absolutely incredible",
        "I forgot to water the plants while we were on vacation",
        "Can you believe how expensive gas prices are these days",
        "My neighbor just got a new puppy and it's very energetic",
        "We should plan a camping trip before summer ends",
        "I finally organized the garage and found my old skateboard",
        "The traffic was terrible this morning because of construction",
        "My sister is moving to a new apartment next month",
        "I baked cookies for the school fundraiser tomorrow",
        "Have you tried that new Thai restaurant on Main Street",
        "We watched the sunset from the rooftop and it was beautiful",
        "I need to schedule an oil change for my car soon",
        "The library book sale has some really great deals",
        "Our garden is producing so many tomatoes this year",
        "I signed up for a pottery class at the community center",
        "The kids want to build a treehouse in the backyard",
        "I just renewed my gym membership for another year",
        "We're having a barbecue this Saturday and everyone's invited",
        "My coworker brought homemade bread to the office today",
        "I've been reading a really great mystery novel lately",
        "The farmers market has fresh peaches this time of year",
        "We adopted a cat from the shelter over the weekend",
        "I need to find a good mechanic for brake repairs",
        "The local band is performing at the festival next week",
        "I'm thinking about taking a cooking class in the fall",
        "Our family is planning a reunion for the holiday season",
    ])

    # ---- Academic / scientific ----
    sentences.extend([
        "The experiment demonstrated a statistically significant correlation",
        "Peer review ensures quality and validity of published research",
        "The hypothesis was tested using a double blind clinical trial",
        "Quantum entanglement enables instantaneous state correlation at distance",
        "The fossil record shows evidence of mass extinction events",
        "Machine learning algorithms require large labeled training datasets",
        "Photosynthesis converts carbon dioxide and water into glucose",
        "The study controlled for demographic and socioeconomic variables",
        "CRISPR gene editing allows precise modifications to DNA sequences",
        "Tectonic plate movement causes earthquakes and volcanic activity",
        "The researcher published findings in a peer reviewed journal",
        "Neural networks learn hierarchical representations of input data",
        "Gravitational waves were first detected by the LIGO observatory",
        "The meta-analysis combined results from forty-two independent studies",
        "Stem cell research offers potential treatments for degenerative diseases",
        "The control group received a placebo instead of the active compound",
        "Dark matter comprises roughly twenty-seven percent of the universe",
        "Behavioral economics studies how psychological factors affect decisions",
        "The peer reviewers recommended major revisions to the manuscript",
        "Protein folding determines the three-dimensional structure and function",
        "Statistical significance was defined at the five percent confidence level",
        "Climate models predict rising sea levels over the next century",
        "The longitudinal study tracked participants over a twenty year period",
        "Nanotechnology enables engineering of materials at the atomic scale",
        "The research grant was funded by the national science foundation",
        "Evolution by natural selection explains adaptation in populations",
        "The systematic review identified gaps in the existing literature",
        "Superconductors carry electric current with zero resistance",
        "The qualitative analysis revealed common themes across interviews",
        "Enzyme catalysis increases the rate of biochemical reactions",
        "The randomized trial enrolled three hundred participants across ten sites",
        "Genomic sequencing technology has become faster and more affordable",
        "The theoretical framework draws on established sociological models",
        "Neuroplasticity allows the brain to reorganize after injury",
        "The survey instrument was validated using a pilot study sample",
    ])

    # ---- Finance / business ----
    sentences.extend([
        "The quarterly earnings report exceeded analyst expectations",
        "Diversifying your investment portfolio reduces overall risk",
        "The company's revenue grew fifteen percent year over year",
        "Interest rates on savings accounts remain historically low",
        "The startup achieved profitability in its third year of operations",
        "Market volatility increased ahead of the federal reserve meeting",
        "The supply chain disruption caused delays across multiple industries",
        "Operating margins improved due to cost reduction initiatives",
        "The acquisition was valued at two billion dollars",
        "Customer retention rates improved after implementing loyalty programs",
        "The budget deficit widened due to increased government spending",
        "Venture capital investment in clean energy reached record levels",
        "The company issued bonds to fund expansion into new markets",
        "Employee stock options vest over a four year schedule",
        "The merger creates the largest company in the telecommunications sector",
        "Accounts receivable turnover ratio indicates collection efficiency",
        "The board declared a quarterly dividend of fifty cents per share",
        "Working capital management is critical for business liquidity",
        "The financial audit found no material misstatements in the reports",
        "Inflation erodes the purchasing power of fixed income investments",
        "The sales pipeline shows strong growth potential for next quarter",
        "Risk management frameworks help identify and mitigate business threats",
        "The initial public offering was oversubscribed by three times",
        "Cash flow from operations turned positive for the first time",
        "The credit rating agency downgraded the sovereign debt outlook",
        "Real estate investment trusts provide exposure to property markets",
        "The fiscal year ends on the thirty-first of December",
        "Total shareholder return outperformed the benchmark index",
        "The cost-benefit analysis supports the proposed capital expenditure",
        "Foreign exchange fluctuations impact multinational company earnings",
        "The balance sheet shows a strong equity position relative to debt",
        "Revenue recognition standards require completion of performance obligations",
        "The break-even analysis determines the minimum sales volume needed",
        "Depreciation expenses reduce taxable income for capital equipment",
        "The pension fund is fully funded according to actuarial estimates",
    ])

    # ---- Cooking / recipes ----
    sentences.extend([
        "Preheat the oven to three hundred and fifty degrees Fahrenheit",
        "Marinate the chicken in soy sauce ginger and garlic overnight",
        "Fold the egg whites gently into the batter to keep it fluffy",
        "Simmer the tomato sauce on low heat for at least thirty minutes",
        "Season the steak with salt and pepper before searing on high heat",
        "Dice the onions and saute them until they become translucent",
        "Let the bread dough rise in a warm place for about one hour",
        "Whisk together the flour baking powder and sugar in a large bowl",
        "Roast the vegetables at high temperature until caramelized",
        "Deglaze the pan with white wine to create a flavorful sauce",
        "Chill the pie crust in the refrigerator before rolling it out",
        "Toast the spices in a dry skillet to release their aroma",
        "Blanch the green beans in boiling water then transfer to ice bath",
        "Reduce the balsamic vinegar until it becomes thick and syrupy",
        "Knead the pasta dough until it is smooth and elastic",
        "Garnish the dish with fresh herbs and a squeeze of lemon",
        "Cream the butter and sugar together until light and fluffy",
        "Strain the broth through a fine mesh sieve for clarity",
        "The sourdough starter needs to be fed daily with flour and water",
        "Let the cake cool completely before applying the frosting",
        "Toss the salad with olive oil and vinegar dressing",
        "Braise the short ribs in red wine for three hours until tender",
        "Zest the lemon directly over the batter for maximum freshness",
        "Temper the chocolate by slowly raising and lowering the temperature",
        "Add the cream cheese to the mixture and blend until smooth",
        "Score the duck breast skin in a crosshatch pattern before cooking",
        "The risotto requires constant stirring while adding broth gradually",
        "Proof the yeast in warm water with a pinch of sugar",
        "Caramelize the sugar in a heavy saucepan over medium heat",
        "Layer the lasagna with pasta sauce ricotta and mozzarella cheese",
    ])

    # ---- Travel / hospitality ----
    sentences.extend([
        "The hotel offers complimentary breakfast and airport shuttle service",
        "Check-in time is three in the afternoon and checkout is at noon",
        "The resort features an infinity pool overlooking the ocean",
        "All rooms include free high-speed wireless internet access",
        "The guided walking tour covers the historic city center landmarks",
        "Flight delays are expected due to severe weather conditions",
        "The cruise itinerary includes stops at five Caribbean islands",
        "Passport renewal takes approximately six to eight weeks",
        "The vacation rental accommodates up to eight guests comfortably",
        "Travel insurance covers trip cancellation and medical emergencies",
        "The boutique hotel is located within walking distance of the beach",
        "The airline offers extra legroom seats for a small additional fee",
        "Our concierge can arrange restaurant reservations and event tickets",
        "The hiking trail is moderate difficulty and takes about four hours",
        "Luggage allowance includes one carry-on and one checked bag",
        "The bed and breakfast serves locally sourced organic meals",
        "Car rental rates include unlimited mileage and basic insurance",
        "The museum admission is free on the first Sunday of each month",
        "The spa offers massage therapy and facial treatment packages",
        "Public transportation passes are available for daily or weekly use",
        "The national park requires advance reservations during peak season",
        "Airport lounges provide comfortable seating and refreshments",
        "The scenic train route passes through mountain tunnels and valleys",
        "Currency exchange services are available at the main terminal",
        "The all-inclusive package covers meals drinks and activities",
        "Local festivals celebrate traditional music food and crafts",
        "The diving excursion explores coral reefs and underwater caves",
        "Room service is available twenty-four hours a day",
        "The bicycle rental shop provides helmets and route maps",
        "Travel advisories recommend checking entry requirements before departure",
    ])

    # ---- Education ----
    sentences.extend([
        "The curriculum includes both theoretical and hands-on coursework",
        "Students must complete a capstone project before graduation",
        "The professor holds office hours every Tuesday and Thursday afternoon",
        "The scholarship covers tuition fees and a monthly living stipend",
        "Group projects develop teamwork and communication skills",
        "The online learning platform offers self-paced video lectures",
        "Academic integrity policies prohibit plagiarism and cheating",
        "The library provides access to thousands of digital journals",
        "Standardized test scores are one factor in admissions decisions",
        "The internship program connects students with industry employers",
        "The teaching assistant leads weekly discussion sections",
        "Foreign language requirements can be met through proficiency exams",
        "The dean's list recognizes students with outstanding academic records",
        "Campus housing includes dormitories and apartment-style residences",
        "The research lab offers undergraduate students hands-on experience",
        "Continuing education courses help professionals update their skills",
        "The seminar focuses on primary source analysis and critical thinking",
        "Financial aid applications are due by the first of March",
        "The school district adopted a new mathematics curriculum",
        "Student organizations provide leadership opportunities outside class",
        "The grading rubric evaluates content organization and presentation",
        "Tutoring services are free for enrolled students in all subjects",
        "The commencement ceremony takes place in the outdoor amphitheater",
        "Exchange programs allow students to study abroad for one semester",
        "The early childhood education program emphasizes play-based learning",
        "Course evaluations help improve teaching quality each semester",
        "The vocational training program prepares students for skilled trades",
        "Academic advisors help students plan their course schedules",
        "The dissertation committee reviews and approves the research proposal",
        "Distance learning programs have expanded access to higher education",
    ])

    # ---- Environment / sustainability ----
    sentences.extend([
        "Recycling programs divert waste from landfills and conserve resources",
        "Solar panel installations have become more affordable for homeowners",
        "The endangered species protection plan covers critical habitat areas",
        "Carbon offset programs fund reforestation and clean energy projects",
        "Plastic pollution threatens marine ecosystems and wildlife health",
        "The water conservation campaign encourages shorter showers",
        "Organic farming practices avoid synthetic pesticides and fertilizers",
        "The wetland restoration project aims to improve local water quality",
        "Electric buses reduce emissions in urban public transportation systems",
        "The company pledged to achieve net zero carbon emissions by 2035",
        "Composting food waste produces nutrient-rich soil for gardens",
        "Wind farms generate clean electricity without burning fossil fuels",
        "The river cleanup effort removed tons of debris and invasive plants",
        "Energy-efficient appliances lower household electricity consumption",
        "Biodiversity loss accelerates when natural habitats are fragmented",
        "The community garden provides fresh produce to local food banks",
        "Green building certification requires meeting strict sustainability standards",
        "Ocean acidification threatens coral reef ecosystems worldwide",
        "The zero-waste movement aims to minimize consumer product disposal",
        "Rainwater harvesting systems supplement household water supplies",
        "Deforestation contributes to both carbon emissions and habitat loss",
        "The environmental impact assessment evaluates effects of the proposed project",
        "Reusable containers reduce single-use packaging in food service",
        "The cap-and-trade system puts a price on industrial carbon emissions",
        "Urban tree planting improves air quality and reduces heat island effects",
        "The fishery management plan sets sustainable catch limits for each season",
        "Geothermal energy harnesses heat from within the earth's crust",
        "The pollution monitoring station measures air quality around the clock",
        "Sustainable fashion brands use recycled and ethically sourced materials",
        "The national wildlife refuge protects migratory bird nesting grounds",
    ])

    # ---- Sports / fitness ----
    sentences.extend([
        "The marathon training plan gradually increases weekly mileage",
        "Proper warm-up exercises help prevent muscle injuries during workouts",
        "The basketball team won the championship in overtime",
        "Swimming is a low-impact exercise that works the entire body",
        "The coach emphasized defensive positioning during practice drills",
        "Protein intake after exercise supports muscle recovery and growth",
        "The tennis match went to five sets before a winner emerged",
        "Interval training alternates between high and low intensity periods",
        "The soccer league starts registration for the fall season",
        "Stretching improves flexibility and reduces post-workout soreness",
        "The cycling route follows scenic trails along the river path",
        "Rest days are essential for muscle repair and performance gains",
        "The referee issued a yellow card for the dangerous tackle",
        "Cross-training helps athletes develop well-rounded fitness levels",
        "The weightlifting program focuses on compound movements for strength",
        "Hydration is crucial during endurance activities in hot weather",
        "The track and field meet features sprints hurdles and relay races",
        "Yoga practice improves balance coordination and mental focus",
        "The golf tournament raised money for the children's hospital",
        "Heart rate zones help athletes train at the appropriate intensity",
        "The rowing team practices early morning sessions on the lake",
        "Foam rolling helps release muscle tension and improve recovery",
        "The ice hockey season runs from October through April",
        "Proper running form reduces the risk of knee and ankle injuries",
        "The climbing gym offers routes for beginners through advanced climbers",
        "Team sports teach valuable lessons about cooperation and communication",
        "The triathlon consists of swimming cycling and running segments",
        "Progressive overload is the principle of gradually increasing training load",
        "The boxing class combines fitness training with self-defense techniques",
        "Recovery nutrition should include both carbohydrates and protein",
    ])

    # ---- Real estate ----
    sentences.extend([
        "The three bedroom house features hardwood floors throughout",
        "The property includes a detached two-car garage and fenced yard",
        "Open floor plan connects the kitchen dining and living areas",
        "The condo association fee covers landscaping pool and gym access",
        "The home inspection revealed minor issues with the roof shingles",
        "The listing price is below the appraised market value",
        "New construction homes come with a ten-year structural warranty",
        "The apartment complex offers in-unit washer dryer connections",
        "Walking distance to schools parks and public transportation",
        "The basement has been finished and can serve as a home office",
        "Energy-efficient windows reduce heating and cooling costs significantly",
        "The property sits on a half-acre lot with mature landscaping",
        "The real estate agent scheduled five showings for the weekend",
        "The kitchen was recently renovated with granite countertops",
        "The mortgage pre-approval letter strengthens your offer position",
        "The neighborhood has low crime rates and excellent school ratings",
        "Townhomes offer a balance between single-family and condo living",
        "The seller agreed to cover closing costs as part of negotiations",
        "The HOA rules restrict exterior paint colors and fence heights",
        "The attic has potential for conversion into additional living space",
        "Commercial property zoning allows both retail and office use",
        "The home warranty covers major appliances and system breakdowns",
        "The property tax assessment increased by three percent this year",
        "The lot offers panoramic mountain views from the rear deck",
        "First-time homebuyer programs offer down payment assistance grants",
    ])

    # ---- Emotional / sentiment-rich ----
    sentences.extend([
        "I am extremely frustrated with the level of service I received",
        "This is the best purchase I have ever made in my entire life",
        "The customer service representative was incredibly helpful and patient",
        "I am deeply disappointed by the quality of this product",
        "What an absolutely wonderful experience from start to finish",
        "I feel completely lost and don't know where to turn for help",
        "The team went above and beyond to solve my problem quickly",
        "I regret buying this and wish I had chosen a different option",
        "The support I received restored my faith in this company",
        "I have never been so angry about a billing mistake before",
        "The atmosphere of the restaurant was warm and welcoming",
        "I'm worried that my complaint is not being taken seriously",
        "Thank you so much for making this process easy and stress-free",
        "The delay is unacceptable and has caused significant inconvenience",
        "I'm thrilled with the results and would recommend this to everyone",
        "This experience has left me feeling undervalued as a customer",
        "The kindness of the staff made a difficult situation much easier",
        "I'm confused by the instructions and need someone to walk me through",
        "The quality exceeded my expectations in every possible way",
        "I feel ignored and my repeated requests have gone unanswered",
        "I appreciate the quick response and thorough explanation provided",
        "The constant issues have eroded my trust in this brand",
        "I'm delighted with the prompt resolution of my concern",
        "The lack of communication throughout the process was very stressful",
        "I'm grateful for the patience shown during my lengthy phone call",
        "My satisfaction with this company has dropped dramatically",
        "The warm welcome we received made us feel like valued guests",
        "I'm overwhelmed by the complexity of the billing statement",
        "Outstanding work by the entire team on this project delivery",
        "I feel let down after years of being a loyal customer",
    ])

    # ---- Short / terse queries (simulate search queries) ----
    sentences.extend([
        "password reset not working",
        "refund request denied",
        "shipping delay update",
        "cancel my subscription",
        "how to export data",
        "payment failed error",
        "change billing address",
        "account suspended why",
        "upgrade plan options",
        "download invoice PDF",
        "missing order tracking",
        "warranty claim process",
        "contact phone number",
        "delete my account",
        "transfer ownership request",
        "discount code expired",
        "mobile app crashing",
        "login page error",
        "data privacy settings",
        "service outage status",
        "installation guide steps",
        "compatibility requirements",
        "trial period extension",
        "bulk pricing available",
        "integration documentation",
        "feature request submit",
        "requesting new functionality",
        "wish list for product updates",
        "suggest a feature improvement",
        "product enhancement request",
        "can you add this feature",
        "we need new capabilities",
        "feature suggestion for the product",
        "please add this to the roadmap",
        "reporting dashboard access",
        "notification preferences",
        "team member invitation",
        "custom domain setup",
        "API rate limit exceeded",
        "backup and restore",
        "multi-language support",
        "single sign-on setup",
        "compliance certification",
    ])

    # ---- Paraphrase pairs (helps W learn semantic equivalence) ----
    sentences.extend([
        # Pair 1: different words, same meaning
        "The computer is running very slowly today",
        "My PC has terrible performance right now",
        # Pair 2
        "The customer is unhappy with their purchase",
        "The buyer is dissatisfied with the product they bought",
        # Pair 3
        "Please send me the document by email",
        "Can you forward the file to my inbox",
        # Pair 4
        "The meeting has been postponed to next week",
        "The conference call was rescheduled for the following week",
        # Pair 5
        "I need to fix this bug before the release",
        "This defect must be resolved prior to deployment",
        # Pair 6
        "The restaurant was fully booked for the evening",
        "There were no available tables at the dining establishment tonight",
        # Pair 7
        "Revenue increased significantly over the quarter",
        "Income rose substantially during the three-month period",
        # Pair 8
        "The patient is experiencing chest pain and shortness of breath",
        "The individual reports thoracic discomfort and difficulty breathing",
        # Pair 9
        "Students must submit assignments before the deadline",
        "Learners are required to turn in their work by the due date",
        # Pair 10
        "The weather forecast predicts rain throughout the weekend",
        "Meteorologists expect precipitation for the next few days",
        # Pair 11
        "The software update includes several security fixes",
        "The patch addresses multiple vulnerability issues",
        # Pair 12
        "I would like to cancel my subscription immediately",
        "Please terminate my membership right away",
        # Pair 13
        "The building needs major structural repairs",
        "The structure requires significant renovation work",
        # Pair 14
        "Children learn best through interactive play activities",
        "Kids acquire knowledge most effectively via hands-on engagement",
        # Pair 15
        "The company laid off hundreds of workers last month",
        "The firm terminated employment for many staff members recently",
        # Pair 16
        "The train arrives at the station every thirty minutes",
        "The rail service departs from the platform at half-hour intervals",
        # Pair 17
        "The medication should be taken with a full glass of water",
        "The drug must be consumed alongside adequate fluid intake",
        # Pair 18
        "The athlete broke the world record in the sprint event",
        "The runner set a new global best time in the race",
        # Pair 19
        "We need to reduce our carbon footprint significantly",
        "Our greenhouse gas emissions must decrease substantially",
        # Pair 20
        "The jury reached a unanimous verdict of not guilty",
        "All members of the panel agreed on an acquittal decision",
    ])

    # ---- Emerging technology ----
    sentences.extend([
        "Artificial intelligence is transforming the healthcare industry",
        "The autonomous vehicle navigated city streets without intervention",
        "Blockchain technology enables decentralized transaction recording",
        "Cloud computing provides scalable resources on demand",
        "Cybersecurity threats are becoming more sophisticated each year",
        "Data analytics helps businesses make informed strategic decisions",
        "Edge computing processes data closer to where it is generated",
        "The Internet of Things connects billions of devices worldwide",
        "Natural language processing enables machines to understand human text",
        "Quantum computing promises exponential speedups for certain problems",
        "Robotic process automation reduces manual work in business operations",
        "Virtual reality creates immersive experiences for training and entertainment",
        "Augmented reality overlays digital information on the physical world",
        "Three-dimensional printing enables rapid prototyping of complex designs",
        "Biometric authentication uses unique physical characteristics for security",
        "The digital twin creates a virtual replica of physical systems",
        "Federated learning trains models without sharing raw data",
        "Generative AI creates new content from learned patterns",
        "The knowledge graph connects related information entities",
        "Low-code platforms enable rapid application development",
        "Predictive maintenance reduces equipment downtime in factories",
        "Recommendation engines personalize content based on user preferences",
        "Sentiment analysis determines the emotional tone of written text",
        "Speech recognition converts spoken language into written text",
        "Transfer learning applies knowledge from one task to another",
        "The warehouse management system tracks inventory in real time",
        "Workflow automation streamlines repetitive business processes",
        "Zero-trust security requires verification for every access request",
        "The API gateway manages and secures microservice communications",
        "Continuous integration ensures code changes are tested automatically",
        "DevOps practices bridge the gap between development and operations",
        "The event-driven architecture reacts to state changes in real time",
        "Infrastructure as code manages cloud resources through configuration files",
        "The monitoring dashboard displays system health metrics and alerts",
        "Service mesh manages communication between distributed services",
    ])

    # ---- Nature / geography ----
    sentences.extend([
        "The climate in coastal regions is moderated by ocean currents",
        "Mountain ecosystems support unique biodiversity at different elevations",
        "Desert plants have adapted to survive with minimal rainfall",
        "Tropical rainforests contain the greatest diversity of plant species",
        "Arctic ice coverage has been declining over recent decades",
        "Coral reefs are often called the rainforests of the sea",
        "The tidal patterns along the coast are influenced by the moon",
        "Volcanic islands form as tectonic plates move over hot spots",
        "River deltas are fertile areas formed by sediment deposits",
        "The permafrost layer is thawing at an unprecedented rate",
        "Mangrove forests protect coastlines from erosion and storm damage",
        "The savanna grasslands support large populations of grazing animals",
        "Glacial lakes formed when ice age glaciers carved out basins",
        "The ocean floor contains vast mountain ranges and deep trenches",
        "Migratory birds travel thousands of miles between seasonal habitats",
        "Underground cave systems can extend for hundreds of kilometers",
        "Seasonal monsoon rains are essential for agriculture in many regions",
        "The continental shelf extends from the shoreline into deeper waters",
        "Barrier islands shift position over time due to wave action and storms",
        "The boreal forest spans the northern latitudes across multiple continents",
    ])

    # ---- Arts / crafts / culture ----
    sentences.extend([
        "The pottery workshop teaches both wheel throwing and hand building",
        "Watercolor painting requires understanding of color transparency",
        "The photography exhibit showcases work from emerging artists",
        "Woodworking projects range from simple shelves to intricate furniture",
        "The knitting pattern includes instructions for beginners and experts",
        "The novel explores themes of identity and belonging in modern society",
        "Poetry workshops encourage creative expression through language",
        "The film uses nonlinear storytelling to build suspense and mystery",
        "The theater production features a cast of local community actors",
        "Digital illustration combines traditional art skills with modern tools",
        "The sculpture garden features works from international contemporary artists",
        "Calligraphy practice develops patience and attention to letter forms",
        "The jazz ensemble performs original compositions every Friday evening",
        "Ceramic glazes react differently depending on the firing temperature",
        "The documentary filmmaker spent three years following the subject",
        "The string quartet will perform selections from classical and modern repertoire",
        "Screen printing allows artists to reproduce designs on fabric and paper",
        "The ballet company premieres a new production inspired by folk tales",
        "Stained glass windows filter light into vibrant colored patterns",
        "The literary magazine publishes short fiction essays and poetry",
        "Mixed media collage combines photographs fabric and found objects",
        "The opera singer trained for ten years before performing professionally",
        "Mosaic art assembles small tiles into larger decorative patterns",
        "The stand-up comedy show features both seasoned and new performers",
        "Landscape architecture blends functional design with natural beauty",
    ])

    # ---- Automotive / transportation ----
    sentences.extend([
        "The hybrid vehicle switches between electric and gasoline power",
        "Regular oil changes extend the life of the engine significantly",
        "The public transit authority expanded bus routes to underserved areas",
        "Tire pressure should be checked monthly for safety and fuel efficiency",
        "The freight rail network moves goods across the country efficiently",
        "Collision detection systems automatically apply the brakes when needed",
        "The carpool lane reduces congestion during peak commute hours",
        "Electric scooter sharing programs provide last-mile transportation options",
        "The mechanic diagnosed the problem as a faulty alternator",
        "Pedestrian crossing signals need to allow sufficient time for all ages",
        "The motorcycle requires a different license endorsement to operate",
        "Speed governors limit the maximum velocity of commercial trucks",
        "Traffic signal timing optimization reduces wait times at intersections",
        "The ferry service connects the island community to the mainland",
        "Anti-lock braking systems prevent wheel lockup during emergency stops",
        "The bicycle lane is separated from vehicle traffic by a concrete barrier",
        "Vehicle emissions testing is required annually in many states",
        "The suspension system absorbs shocks from uneven road surfaces",
        "Ride-sharing services have changed urban transportation patterns",
        "The transmission fluid should be replaced according to the maintenance schedule",
        "High-speed rail offers an alternative to short-distance flights",
        "The parking structure has electric vehicle charging stations on every level",
        "Dashboard warning lights indicate when maintenance is required",
        "Snow tires provide better traction on icy winter roads",
        "The subway system carries millions of passengers every day",
    ])

    # ---- Food / agriculture ----
    sentences.extend([
        "Crop rotation prevents soil depletion and reduces pest problems",
        "The food safety inspection covers storage temperatures and hygiene practices",
        "Heritage breed livestock are valued for their genetic diversity",
        "Vertical farming allows food production in urban environments year-round",
        "The winery produces small-batch wines from locally grown grapes",
        "Irrigation systems distribute water efficiently to agricultural fields",
        "The farmers cooperative negotiates better prices for member producers",
        "Genetically modified crops are designed to resist specific pests",
        "The bakery sources flour from regional grain mills",
        "Aquaculture farms raise fish and shellfish in controlled environments",
        "Food preservation methods include canning freezing and dehydration",
        "The olive harvest takes place in late autumn when fruit is ripe",
        "Sustainable fishing practices protect marine populations from depletion",
        "The cheese aging process develops complex flavors over several months",
        "Pollinator gardens attract bees butterflies and other beneficial insects",
        "The brewery crafts seasonal beers using locally sourced hops and barley",
        "Soil testing reveals nutrient levels and guides fertilizer application",
        "Free-range poultry have access to outdoor areas during the day",
        "The spice market offers dried herbs and seasonings from around the world",
        "Precision agriculture uses satellite data to optimize planting decisions",
        "The chocolate maker sources cacao beans directly from small farms",
        "Hydroponics grows plants in nutrient-rich water without soil",
        "The food truck serves fusion cuisine combining Korean and Mexican flavors",
        "Cold chain logistics ensure perishable goods maintain proper temperature",
        "The tea plantation occupies terraced hillsides at high elevation",
    ])

    # ---- Home improvement / DIY ----
    sentences.extend([
        "Apply painter's tape along edges for clean straight paint lines",
        "The plumber fixed the leaking faucet by replacing the worn washer",
        "Insulation in the attic reduces heating costs during winter months",
        "The electrician installed additional outlets in the home office",
        "Grout between tiles should be sealed to prevent moisture damage",
        "The deck boards need to be stained and sealed every two years",
        "A level ensures shelves are mounted perfectly horizontal on the wall",
        "The garbage disposal unit requires periodic cleaning with ice and salt",
        "Weather stripping around doors and windows prevents drafts and air leaks",
        "The water heater should be flushed annually to remove sediment buildup",
        "Crown molding adds an elegant transition between walls and ceilings",
        "The smoke detectors need fresh batteries at least once per year",
        "Caulking around the bathtub prevents water from damaging the subfloor",
        "The gutters must be cleaned before the rainy season to prevent overflow",
        "Laminate flooring installation requires an underlayment for moisture protection",
        "The circuit breaker trips when too many appliances run simultaneously",
        "A programmable thermostat automates temperature adjustments throughout the day",
        "The sump pump activates automatically when water levels rise in the basement",
        "Pressure washing removes dirt and mildew from exterior siding surfaces",
        "The garage door opener needs lubrication on the chain and rollers",
        "Replacing old windows with double-pane glass improves energy efficiency",
        "The landscaping project includes a stone retaining wall and flower beds",
        "Cabinet hardware replacement gives the kitchen a fresh updated look",
        "Pipe insulation prevents frozen pipes during extreme cold weather",
        "The bathroom renovation includes new tile flooring and a glass shower enclosure",
    ])

    # ---- Pet care / veterinary ----
    sentences.extend([
        "Annual vaccinations protect pets from common infectious diseases",
        "The veterinarian recommended a dental cleaning for the older cat",
        "Flea and tick prevention should be applied monthly during warm months",
        "The shelter has many adoptable dogs and cats looking for homes",
        "Puppy training classes teach basic obedience commands and socialization",
        "The aquarium filter needs cleaning every two weeks for water quality",
        "Senior pets benefit from more frequent veterinary checkups",
        "The bird cage should be large enough for the parrot to spread its wings",
        "Grain-free diets may be appropriate for pets with food allergies",
        "The dog walker takes three different routes to keep walks interesting",
        "Spaying or neutering pets helps control the population of strays",
        "The reptile enclosure requires specific temperature and humidity levels",
        "Microchipping pets increases the chances of reunion if they get lost",
        "The grooming salon offers baths haircuts and nail trimming services",
        "Separation anxiety in dogs can be managed with gradual desensitization",
        "The horse stable provides daily turnout in the paddock area",
        "Raw food diets for pets should be formulated with veterinary guidance",
        "The cat scratching post saves furniture from claw damage",
        "Pet insurance covers unexpected veterinary bills and emergency treatments",
        "The rabbit hutch needs fresh bedding and hay replaced regularly",
        "Heartworm prevention medication should be given year-round in many climates",
        "The pet boarding facility provides webcam access so owners can check in",
        "Dental chews help reduce tartar buildup on dogs teeth between cleanings",
        "The tropical fish tank requires stable water temperature and pH levels",
        "Positive reinforcement training uses treats and praise to encourage behavior",
    ])

    # ---- Psychology / mental wellness ----
    sentences.extend([
        "Mindfulness meditation reduces stress and improves emotional regulation",
        "Cognitive behavioral therapy helps patients identify negative thought patterns",
        "Adequate sleep is essential for memory consolidation and mental clarity",
        "Social connections are a key predictor of long-term psychological wellbeing",
        "Journaling can help process difficult emotions and track personal growth",
        "Burnout results from prolonged exposure to chronic workplace stress",
        "Gratitude practices are associated with increased life satisfaction",
        "The grief counselor supports families through the stages of bereavement",
        "Physical exercise releases endorphins that naturally improve mood",
        "Setting healthy boundaries is important for maintaining relationships",
        "Screen time before bed interferes with natural sleep hormone production",
        "The psychologist administered a standardized personality assessment",
        "Self-compassion involves treating yourself with the same kindness as a friend",
        "Exposure therapy gradually reduces the fear response to specific triggers",
        "Work-life balance requires intentional boundaries between professional and personal time",
        "The support group meets weekly to share experiences and coping strategies",
        "Emotional intelligence involves recognizing and managing your own emotions",
        "The relaxation technique involves progressive tensing and releasing of muscles",
        "Childhood attachment patterns influence adult relationship dynamics",
        "Digital detox periods can reduce anxiety and improve present-moment awareness",
        "The crisis helpline provides immediate support for people in emotional distress",
        "Resilience is the ability to recover from adversity and adapt to change",
        "Art therapy uses creative expression as a pathway to emotional healing",
        "Imposter syndrome affects many high-achieving professionals and students",
        "The therapist uses motivational interviewing to support behavior change",
    ])

    # ---- Manufacturing / engineering ----
    sentences.extend([
        "The assembly line produces three hundred units per hour at full capacity",
        "Quality control inspectors check products at multiple stages of production",
        "Computer-aided design software creates precise three-dimensional models",
        "The welding robot achieves consistent joint quality at high speed",
        "Lean manufacturing principles eliminate waste and improve efficiency",
        "The factory operates three shifts to maintain continuous production",
        "Material specifications define the exact composition required for each part",
        "The hydraulic press applies thousands of pounds of force to shape metal",
        "Supply chain management coordinates the flow of materials to the factory",
        "The CNC machine cuts parts with micrometer-level precision",
        "Statistical process control monitors manufacturing quality in real time",
        "The injection molding machine produces plastic components rapidly",
        "Safety protocols require protective equipment in all production areas",
        "The conveyor belt system moves products between workstations efficiently",
        "Failure mode analysis identifies potential weaknesses before they cause problems",
        "The calibration schedule ensures measurement instruments remain accurate",
        "Just-in-time inventory reduces storage costs and minimizes waste",
        "The heat treatment process hardens the steel components for durability",
        "Additive manufacturing builds parts layer by layer from digital files",
        "The packaging line wraps seals and labels products for distribution",
        "Preventive maintenance schedules reduce unplanned equipment downtime",
        "The electrical panel distributes power to all machines on the factory floor",
        "Tolerance stacking analysis ensures assembled parts fit together correctly",
        "The paint booth applies an even coat under controlled temperature conditions",
        "ISO certification demonstrates adherence to international quality standards",
    ])

    # ---- Music / performing arts ----
    sentences.extend([
        "The guitarist practiced scales for two hours to build finger dexterity",
        "The orchestra rehearsal focused on the dynamics of the second movement",
        "Digital audio workstations enable music production from a home studio",
        "The choir performs sacred and secular repertoire from multiple centuries",
        "Sound mixing adjusts the volume balance between instruments and vocals",
        "The piano tuner adjusts string tension to achieve proper pitch",
        "Vinyl records have experienced a resurgence among music collectors",
        "The music festival lineup features artists from genres spanning jazz to electronic",
        "Ear training exercises develop the ability to identify intervals and chords",
        "The drum machine provides a consistent beat for recording sessions",
        "Song lyrics express universal themes of love loss and human experience",
        "The concert venue has capacity for two thousand seated attendees",
        "Music therapy helps patients with neurological conditions improve motor function",
        "The recording studio uses acoustic treatment panels to control sound reflections",
        "Sight-reading skills allow musicians to perform unfamiliar pieces quickly",
        "The bass player anchors the rhythm section with steady groove patterns",
        "Streaming platforms have fundamentally changed how listeners discover new music",
        "The conductor shapes the interpretation through tempo and dynamic choices",
        "Musical notation provides a standardized system for representing compositions",
        "The singer-songwriter performs original material at open mic nights",
        "Audio mastering is the final step before a recording is released",
        "The marching band performs choreographed routines during halftime shows",
        "Acoustic guitar techniques include fingerpicking strumming and tapping",
        "The music teacher adapts lesson plans to each student's skill level",
        "Synthesizers create electronic sounds through oscillators filters and amplifiers",
    ])

    # ---- History / social studies ----
    sentences.extend([
        "The industrial revolution transformed manufacturing from hand to machine production",
        "Ancient civilizations developed along major river systems for water access",
        "The printing press dramatically increased the spread of information and literacy",
        "Colonial trade routes connected distant continents through maritime commerce",
        "The civil rights movement fought for equality and an end to discrimination",
        "Archaeological excavations have uncovered artifacts dating back thousands of years",
        "The Renaissance period saw a revival of interest in classical art and learning",
        "Immigration waves shaped the cultural diversity of growing urban centers",
        "The telegraph revolutionized long-distance communication in the nineteenth century",
        "Medieval castles served as defensive fortifications and centers of governance",
        "The abolition of slavery was a pivotal moment in human rights history",
        "Ancient agricultural practices included irrigation and selective crop breeding",
        "The space race accelerated technological innovation during the cold war era",
        "Oral traditions preserved cultural knowledge before written languages developed",
        "The suffrage movement secured voting rights for women in many democracies",
        "Feudal societies organized around hierarchies of lords vassals and serfs",
        "The silk road facilitated cultural and economic exchange between East and West",
        "Urbanization accelerated as people migrated from farms to factory cities",
        "The Enlightenment promoted reason individual rights and scientific inquiry",
        "World wars of the twentieth century reshaped national borders and alliances",
        "Ancient libraries such as Alexandria collected knowledge from the known world",
        "The labor movement fought for fair wages safe conditions and reasonable hours",
        "Navigation advances enabled long-distance ocean voyages of exploration",
        "The green revolution dramatically increased global food production capacity",
        "Constitutional governments established frameworks for democratic governance",
    ])

    # ---- Parenting / family ----
    sentences.extend([
        "The pediatrician recommends breastfeeding for at least the first six months",
        "Childproofing the home includes covering outlets and securing heavy furniture",
        "Reading aloud to children every day builds vocabulary and a love of books",
        "The daycare center follows a structured schedule of learning and play",
        "Consistent bedtime routines help children fall asleep more easily",
        "The school supplies list includes notebooks pencils and a backpack",
        "Family meals together strengthen bonds and encourage healthy eating habits",
        "The playground equipment meets current safety standards for young children",
        "Positive discipline focuses on teaching rather than punishing behavior",
        "The car seat must be installed rear-facing for infants and small toddlers",
        "Screen time limits help ensure children engage in physical and creative play",
        "The babysitter is certified in infant and child CPR and first aid",
        "Sibling rivalry is a normal part of childhood development",
        "The after-school program offers homework help and supervised recreation",
        "Potty training readiness varies widely between individual children",
        "The family budget includes savings for each child's college education fund",
        "Allergies should be communicated to teachers and school cafeteria staff",
        "The birthday party includes games crafts and a homemade cake",
        "Encouraging independence in age-appropriate tasks builds confidence",
        "The parent-teacher conference provides updates on academic and social progress",
        "Teenagers need clear expectations balanced with increasing autonomy",
        "The summer camp program includes swimming hiking and arts and crafts",
        "New parents often experience sleep deprivation during the first few months",
        "The children's museum has interactive exhibits designed for hands-on learning",
        "Establishing a homework routine helps children develop study habits early",
    ])

    # ---- Insurance / risk ----
    sentences.extend([
        "The auto insurance policy includes collision and comprehensive coverage",
        "Homeowner's insurance protects against fire theft and natural disasters",
        "The deductible is the amount the policyholder pays before coverage begins",
        "Life insurance provides financial protection for beneficiaries and dependents",
        "The claims adjuster assessed the damage and approved the repair estimate",
        "Umbrella policies provide additional liability coverage beyond standard limits",
        "Health insurance premiums vary based on age location and plan type",
        "The flood insurance requirement applies to properties in high-risk zones",
        "Disability insurance replaces income if a worker cannot perform their job",
        "The actuarial tables estimate risk probabilities for pricing calculations",
        "Renters insurance covers personal belongings and liability in leased properties",
        "The insurance broker compares quotes from multiple carriers for the client",
        "Workers compensation covers medical expenses for injuries sustained on the job",
        "The risk assessment identified several vulnerabilities in the business operations",
        "Commercial general liability insurance protects businesses from lawsuit costs",
        "The underwriting process evaluates the risk before issuing a new policy",
        "Pet insurance helps cover unexpected veterinary expenses and surgeries",
        "The grace period allows policyholders extra time to make late premium payments",
        "Professional liability insurance protects against malpractice claims",
        "The insurance company denied the claim due to a policy exclusion clause",
        "Long-term care insurance covers assisted living and nursing home expenses",
        "Catastrophic coverage plans have low premiums but very high deductibles",
        "The subrogation process recovers costs from the party responsible for the loss",
        "Event cancellation insurance reimburses costs if a planned event cannot proceed",
        "The annual premium review may result in rate adjustments based on claims history",
    ])

    # ---- Telecommunications ----
    sentences.extend([
        "The fiber optic network provides gigabit internet speeds to residential customers",
        "Cell tower placement requires balancing coverage area with local regulations",
        "The voice over IP system reduces long-distance calling costs significantly",
        "Bandwidth throttling occurs when network usage exceeds the data cap",
        "The telecommunications company offers bundled phone internet and television plans",
        "Signal strength decreases in areas far from the nearest transmission tower",
        "The router firmware update improved wireless network stability and security",
        "Satellite internet service reaches remote areas without cable infrastructure",
        "The customer upgraded from a basic plan to unlimited data and messaging",
        "Network latency affects the quality of real-time video conferencing calls",
        "The mobile carrier launched its fifth-generation wireless network this year",
        "Unified communications platforms integrate voice video and messaging tools",
        "The service level agreement guarantees a minimum uptime of ninety-nine percent",
        "Electromagnetic interference can disrupt wireless signal quality indoors",
        "The conference bridge supports up to fifty simultaneous participants",
        "The telephone exchange routes calls between local and long-distance networks",
        "Data encryption protects communications transmitted over public networks",
        "The internet service provider offers symmetrical upload and download speeds",
        "Mesh networking extends WiFi coverage to every room in the building",
        "The emergency communication system sends alerts to all mobile devices in the area",
    ])

    # ---- Logistics / supply chain ----
    sentences.extend([
        "The distribution center processes over ten thousand packages every day",
        "Last-mile delivery is the most expensive segment of the shipping process",
        "The barcode scanning system tracks every item from warehouse to customer",
        "Freight consolidation reduces shipping costs by combining smaller loads",
        "The cold storage facility maintains temperatures below freezing year round",
        "Route optimization software reduces fuel consumption and delivery times",
        "The customs broker handles import documentation and regulatory compliance",
        "Pallet racking systems maximize vertical storage space in the warehouse",
        "The third-party logistics provider manages warehousing and transportation",
        "Cross-docking transfers goods directly from inbound to outbound trucks",
        "The shipping manifest lists all items quantities and destination addresses",
        "Demand forecasting algorithms predict inventory needs for the upcoming season",
        "The returns processing center inspects refurbishes and restocks returned items",
        "Intermodal shipping combines truck rail and ocean transport for efficiency",
        "The automated sorting system directs packages to the correct loading dock",
        "Safety stock levels prevent stockouts during unexpected demand surges",
        "The delivery drone can transport small packages within a five-mile radius",
        "Order fulfillment accuracy is measured as a key performance indicator",
        "The supply chain visibility platform shows shipment status in real time",
        "Vendor-managed inventory allows suppliers to monitor and replenish stock levels",
    ])

    # ---- Astronomy / space ----
    sentences.extend([
        "The Hubble Space Telescope has captured detailed images of distant galaxies",
        "A light-year measures the distance light travels in one calendar year",
        "The Mars rover collected rock samples from the surface of the red planet",
        "Solar flares can disrupt satellite communications and power grid systems",
        "The International Space Station orbits Earth approximately every ninety minutes",
        "Black holes have gravitational fields so strong that light cannot escape",
        "The asteroid belt lies between the orbits of Mars and Jupiter",
        "Exoplanet detection methods include the transit technique and radial velocity",
        "The Milky Way galaxy contains an estimated two hundred billion stars",
        "Neutron stars are incredibly dense remnants of massive stellar explosions",
        "The launch window for the planetary probe opens once every twenty-six months",
        "Cosmic microwave background radiation provides evidence of the early universe",
        "The telescope mirror must be ground to nanometer-level precision",
        "Comets develop visible tails as they approach the sun and ice vaporizes",
        "Space debris poses a growing collision risk for operational satellites",
        "The Doppler effect reveals whether a star is moving toward or away from us",
        "Binary star systems consist of two stars orbiting a common center of mass",
        "Astronaut training includes simulated spacewalks in underwater facilities",
        "The habitable zone is the orbital region where liquid water could exist",
        "Dark energy is believed to drive the accelerating expansion of the universe",
    ])

    # ---- Philosophy / ethics ----
    sentences.extend([
        "Utilitarianism evaluates actions based on the greatest good for the greatest number",
        "The trolley problem illustrates tensions between consequentialist and deontological ethics",
        "Free will and determinism have been debated by philosophers for centuries",
        "Epistemology is the branch of philosophy concerned with the nature of knowledge",
        "The social contract theory explains the legitimacy of government authority",
        "Existentialism emphasizes individual freedom responsibility and authentic choice",
        "Moral relativism holds that ethical standards vary across cultures and contexts",
        "The categorical imperative states that one should act only on universalizable principles",
        "Stoic philosophy teaches acceptance of things beyond personal control",
        "The problem of evil questions how suffering coexists with a benevolent creator",
        "Pragmatism evaluates ideas based on their practical consequences and usefulness",
        "Virtue ethics focuses on developing good character traits rather than following rules",
        "The veil of ignorance thought experiment promotes fairness in social institutions",
        "Phenomenology studies the structures of conscious experience from the first person",
        "Bioethics addresses moral questions arising from advances in medicine and biology",
        "The mind-body problem explores the relationship between mental and physical states",
        "Political philosophy examines concepts of justice liberty and the role of the state",
        "Aesthetic theory investigates the nature of beauty art and taste",
        "The is-ought problem concerns deriving moral conclusions from factual premises",
        "Environmental ethics considers the moral relationship between humans and nature",
    ])

    # ---- Additional paraphrase pairs (more semantic equivalences for W) ----
    sentences.extend([
        # Pair 21
        "The server is down and users cannot access the website",
        "The web service is unavailable and customers are unable to reach the site",
        # Pair 22
        "The price of the item was reduced during the sale event",
        "The cost of the product was discounted at the promotional event",
        # Pair 23
        "The manager approved the request for time off next month",
        "The supervisor authorized the leave of absence for the coming month",
        # Pair 24
        "Heavy rainfall caused widespread flooding in the lowland areas",
        "Intense precipitation led to extensive inundation of the valley regions",
        # Pair 25
        "The student received the highest grade in the mathematics class",
        "The pupil earned the top score in the math course",
        # Pair 26
        "The factory produces electronic components for the automotive industry",
        "The manufacturing plant makes electrical parts for car makers",
        # Pair 27
        "The flight was cancelled due to mechanical issues with the aircraft",
        "The airplane trip was called off because of equipment problems with the plane",
        # Pair 28
        "The report must be submitted by the end of the business day",
        "The document needs to be turned in before close of work today",
        # Pair 29
        "The new policy prohibits smoking anywhere on company premises",
        "The updated regulation bans tobacco use throughout the corporate property",
        # Pair 30
        "The elderly resident requires assistance with daily living activities",
        "The senior citizen needs help with everyday tasks and routines",
        # Pair 31
        "The contractor estimated the renovation will take approximately eight weeks",
        "The builder projected the remodeling job would last about two months",
        # Pair 32
        "The charity organization distributed meals to homeless individuals",
        "The nonprofit group provided food to people without permanent shelter",
        # Pair 33
        "Traffic congestion during rush hour adds thirty minutes to the commute",
        "Peak hour gridlock extends the travel time by half an hour",
        # Pair 34
        "The teacher assigned a research paper on climate change effects",
        "The instructor gave students an essay about the impacts of global warming",
        # Pair 35
        "The battery in my phone drains quickly when using navigation apps",
        "My mobile device loses charge fast when running GPS mapping software",
    ])

    # ---- Short queries / search-like phrases (more variety) ----
    sentences.extend([
        "best practices data migration",
        "troubleshoot network connectivity",
        "employee onboarding checklist",
        "annual performance review template",
        "remote work equipment setup",
        "customer feedback survey questions",
        "project timeline gantt chart",
        "security incident response plan",
        "vendor evaluation criteria",
        "change management process steps",
        "disaster recovery procedures",
        "budget variance analysis",
        "user acceptance testing criteria",
        "regulatory compliance checklist",
        "supply chain risk assessment",
        "stakeholder communication plan",
        "software release notes template",
        "database backup schedule",
        "accessibility compliance standards",
        "content marketing strategy guide",
        "inventory management best practices",
        "return on investment calculation",
        "server capacity planning guide",
        "customer journey map template",
        "agile sprint planning guide",
        "email deliverability troubleshooting",
        "cross-functional team charter",
        "sales pipeline report format",
        "employee satisfaction survey results",
        "crisis communication protocol",
        "technical debt prioritization",
        "service desk escalation matrix",
        "data governance framework",
        "digital transformation roadmap",
        "quality assurance test plan",
    ])

    # ---- Mathematics / statistics ----
    sentences.extend([
        "The standard deviation measures the spread of values around the mean",
        "Linear regression models the relationship between dependent and independent variables",
        "The probability of two independent events both occurring is their product",
        "Matrix multiplication requires the inner dimensions to be compatible",
        "The Fourier transform decomposes signals into constituent frequency components",
        "Bayesian inference updates probability estimates as new evidence becomes available",
        "The central limit theorem states that sample means approach a normal distribution",
        "Eigenvalues and eigenvectors reveal important properties of linear transformations",
        "The Pythagorean theorem relates the sides of a right triangle",
        "Calculus enables the computation of rates of change and accumulated quantities",
        "The chi-squared test determines whether observed frequencies differ from expected",
        "Differential equations model systems that change continuously over time",
        "The binomial distribution describes the number of successes in fixed trials",
        "Set theory provides the foundational language for modern mathematics",
        "The traveling salesman problem seeks the shortest route visiting all cities",
        "Monte Carlo simulations use random sampling to estimate complex outcomes",
        "The prime number theorem describes the distribution of primes among integers",
        "Optimization algorithms find the minimum or maximum of objective functions",
        "The law of large numbers ensures averages converge to expected values",
        "Graph theory studies the properties of networks of connected nodes",
    ])

    # ---- Gardening / horticulture ----
    sentences.extend([
        "Tomato plants need at least six hours of direct sunlight each day",
        "Composting kitchen scraps creates nutrient-rich soil amendment for the garden",
        "The rose bushes should be pruned in early spring before new growth begins",
        "Raised garden beds improve drainage and make planting more accessible",
        "Mulching around plants conserves moisture and suppresses weed growth",
        "The herb garden includes basil rosemary thyme and oregano",
        "Perennial flowers return year after year without replanting",
        "Soil pH affects which nutrients are available to plant roots",
        "The drip irrigation system delivers water directly to the base of each plant",
        "Companion planting pairs beneficial species together to improve growth",
        "The greenhouse extends the growing season through temperature control",
        "Seed starting indoors gives transplants a head start before spring planting",
        "Native plants require less water and maintenance than exotic ornamental species",
        "The trellis supports climbing vegetables like beans cucumbers and peas",
        "Dividing mature perennials rejuvenates the plants and creates new specimens",
        "The lawn requires regular mowing fertilizing and aerating for healthy growth",
        "Container gardening allows growing plants on patios and small balconies",
        "The frost date determines when it is safe to plant tender crops outdoors",
        "Beneficial insects like ladybugs prey on aphids and other garden pests",
        "The succulent collection thrives in well-draining sandy soil with minimal water",
    ])

    # ---- Energy / utilities ----
    sentences.extend([
        "The power grid distributes electricity from generating stations to consumers",
        "Smart meters provide real-time energy consumption data to homeowners",
        "The nuclear power plant generates electricity through controlled fission reactions",
        "Peak demand pricing encourages consumers to shift usage to off-peak hours",
        "The natural gas pipeline transports fuel from production fields to distribution centers",
        "Battery storage systems capture renewable energy for use when generation stops",
        "The utility company offers rebates for installing energy-efficient appliances",
        "The hydroelectric dam converts the energy of falling water into electricity",
        "Net metering allows solar panel owners to sell excess power back to the grid",
        "The combined heat and power system achieves higher overall energy efficiency",
        "The voltage transformer steps down high transmission voltage for residential use",
        "Energy audits identify opportunities to reduce consumption and lower bills",
        "The wind turbine generator converts rotational energy into alternating current",
        "Demand response programs reduce load during periods of grid stress",
        "The electrical substation routes power from transmission lines to local feeders",
        "Time-of-use tariffs charge different rates depending on when electricity is consumed",
        "The biogas digester converts organic waste into methane for fuel and power",
        "Grid modernization includes installing sensors and automated switching equipment",
        "The fuel cell generates electricity through an electrochemical reaction",
        "Distributed energy resources include rooftop solar small wind and battery systems",
    ])

    # ---- Government / public policy ----
    sentences.extend([
        "The municipal government approved the annual operating budget for the city",
        "Voter registration deadlines vary by state and election type",
        "The regulatory agency issued new guidelines for food labeling requirements",
        "Public comment periods allow citizens to provide input on proposed regulations",
        "The census data determines the allocation of federal funding and legislative seats",
        "Emergency management plans coordinate response across multiple government agencies",
        "The zoning board reviews applications for land use changes and variances",
        "Tax policy changes affect both individual taxpayers and business entities",
        "The public works department maintains roads bridges and water infrastructure",
        "Campaign finance laws regulate contributions and spending in elections",
        "The social services department administers benefits for eligible residents",
        "Legislative committees hold hearings to gather testimony on proposed bills",
        "The freedom of information act allows citizens to request government records",
        "Public health departments monitor disease outbreaks and coordinate vaccination efforts",
        "The building code establishes minimum standards for construction safety",
        "Executive orders direct the operations of federal agencies and departments",
        "The diplomatic mission negotiates trade agreements between the two countries",
        "Grant programs fund community development projects in underserved areas",
        "The inspector general investigates waste fraud and abuse in government programs",
        "Ballot measures allow voters to decide directly on proposed laws and amendments",
    ])

    # ---- Additional mixed sentences for coverage breadth ----
    sentences.extend([
        "The librarian organized a summer reading challenge for children of all ages",
        "Ocean shipping containers are standardized to fit on trucks trains and ships",
        "The volunteer firefighters responded to the emergency within minutes of the call",
        "Ergonomic workstation design reduces the risk of repetitive strain injuries",
        "The retirement planning seminar covered investment strategies and tax implications",
        "Renewable energy certificates verify that electricity was generated from clean sources",
        "The archaeological survey identified potential excavation sites along the riverbank",
        "Noise pollution from construction affects the quality of life for nearby residents",
        "The literacy program provides free tutoring for adult learners in the community",
        "Infectious disease protocols require isolation and contact tracing procedures",
        "The antique restoration shop specializes in refinishing wooden furniture",
        "Digital signatures provide legally binding authentication for electronic documents",
        "The wildlife sanctuary rehabilitates injured animals for release back into nature",
        "Inventory shrinkage results from theft damage and administrative errors",
        "The apprenticeship program combines classroom instruction with on-the-job training",
        "Water quality testing detects contaminants that could affect public health",
        "The nonprofit board of directors meets quarterly to review financial statements",
        "Biomechanical analysis helps athletes improve technique and prevent injury",
        "The symphony orchestra season includes eight subscription concert programs",
        "Drought-resistant landscaping reduces outdoor water consumption significantly",
        "The data breach notification law requires companies to inform affected individuals",
        "Heritage preservation efforts protect historically significant buildings from demolition",
        "The emergency room triage system prioritizes patients based on severity of condition",
        "Supply and demand dynamics determine the equilibrium price in competitive markets",
        "The mentorship program pairs experienced professionals with early-career employees",
        "Thermal imaging cameras detect heat loss in buildings during energy audits",
        "The cooperative grocery store is owned and operated by its member shoppers",
        "Standardized testing provides a consistent measure of student achievement across schools",
        "The watershed management plan protects drinking water sources from contamination",
        "Assistive technology helps people with disabilities participate fully in daily life",
        "The craft brewery taproom offers flights of seasonal and year-round selections",
        "Carbon fiber composites provide high strength with minimal weight for aerospace use",
        "The community health center offers sliding-scale fees based on patient income",
        "Marine navigation relies on GPS charts and radar for safe vessel operation",
        "The genealogy research traced the family lineage back five generations",
        "Renewable portfolio standards require utilities to source a percentage from clean energy",
        "The debate tournament follows a structured format with timed speeches and rebuttals",
        "Soil erosion control measures include terracing cover crops and windbreaks",
        "The employee assistance program provides confidential counseling and referral services",
        "Acoustic engineering optimizes sound quality in concert halls and recording spaces",
    ])

    # ---- Fashion / textiles ----
    sentences.extend([
        "The designer showcased a collection inspired by Japanese minimalism",
        "Cotton blend fabrics offer comfort and wrinkle resistance for everyday wear",
        "The tailor adjusted the suit jacket to fit the client's shoulders perfectly",
        "Sustainable fashion emphasizes durability over disposable fast fashion trends",
        "The vintage clothing store curates pieces from the nineteen fifties and sixties",
        "Textile dyeing processes use large quantities of water and chemical compounds",
        "The shoe manufacturer uses ethically sourced leather from certified tanneries",
        "Pattern making translates design sketches into templates for cutting fabric",
        "The fashion week runway show attracted buyers from major department stores",
        "Athleisure clothing blurs the line between athletic and casual everyday wear",
        "The denim factory produces raw selvedge jeans on traditional shuttle looms",
        "Silk production involves harvesting fibers from silkworm cocoons",
        "The accessory line includes handcrafted belts scarves and jewelry pieces",
        "Fast fashion supply chains prioritize speed and cost over environmental impact",
        "The uniform supplier provides embroidered workwear for corporate clients",
        "Technical fabrics with moisture-wicking properties keep athletes cool and dry",
        "The hat maker crafts custom millinery for weddings and formal occasions",
        "Organic cotton certification ensures no synthetic chemicals were used in growing",
        "The footwear company designs ergonomic shoes for healthcare professionals",
        "Upcycling transforms discarded materials into new clothing and accessories",
    ])

    # ---- Marine / nautical ----
    sentences.extend([
        "The harbor master coordinates vessel traffic entering and leaving the port",
        "Life jackets must be worn by all passengers on small recreational boats",
        "The cargo ship transports goods across the Pacific Ocean in forty days",
        "Tidal charts help sailors plan departures and arrivals around water depths",
        "The marine biologist studies the migration patterns of humpback whales",
        "Boat hull maintenance includes cleaning scraping and applying antifouling paint",
        "The lighthouse guides ships away from rocky shoals during poor visibility",
        "Fishing regulations set seasonal limits to protect spawning populations",
        "The yacht club offers sailing lessons for children and adults every summer",
        "Navigation buoys mark safe channels and warn of submerged hazards",
        "The coast guard conducted a search and rescue operation after the distress call",
        "Desalination plants convert seawater into fresh water for coastal communities",
        "The submarine descends to depths of several hundred meters for research purposes",
        "Port congestion delays cause ripple effects throughout the global supply chain",
        "The kayaking excursion explores sheltered coves and mangrove waterways",
        "Marine pollution regulations restrict the discharge of waste from vessels at sea",
        "The tide pool reveals small creatures adapted to the intertidal environment",
        "Ship ballast water management prevents the spread of invasive aquatic species",
        "The sailing regatta attracts competitive racers from around the region",
        "Underwater sonar technology maps the ocean floor topography in detail",
    ])

    # ---- Human resources ----
    sentences.extend([
        "The job posting requires five years of experience and a relevant degree",
        "The onboarding process includes orientation paperwork and system access setup",
        "Performance reviews are conducted annually with mid-year progress check-ins",
        "The employee handbook outlines company policies on conduct and benefits",
        "Diversity and inclusion training is mandatory for all management staff",
        "The exit interview gathers feedback from departing employees about their experience",
        "Payroll processing must be completed by the fifteenth of each month",
        "The benefits package includes health insurance retirement plan and paid time off",
        "The hiring manager conducted three rounds of interviews for the senior position",
        "The workplace harassment policy provides clear reporting and investigation procedures",
        "Professional development budgets allow employees to attend conferences and courses",
        "The recruitment agency specializes in placing candidates in technology roles",
        "Overtime compensation applies when employees work more than forty hours per week",
        "The talent acquisition team uses applicant tracking software to manage candidates",
        "The compensation study ensures salaries are competitive with market benchmarks",
        "Employee recognition programs celebrate achievements and boost workplace morale",
        "The succession planning process identifies future leaders within the organization",
        "The remote work policy defines expectations for availability and communication",
        "Background checks are completed before extending an official offer of employment",
        "The company wellness program includes gym membership discounts and health screenings",
    ])

    # ---- Architecture / urban planning ----
    sentences.extend([
        "The architect designed the building to maximize natural light through large windows",
        "Mixed-use development combines residential commercial and retail spaces in one area",
        "Building codes require emergency exits to be clearly marked and unobstructed",
        "The urban planner proposed adding green spaces throughout the downtown corridor",
        "Load-bearing walls cannot be removed without structural engineering assessment",
        "The city master plan envisions pedestrian-friendly neighborhoods connected by transit",
        "The building's facade features a combination of glass steel and reclaimed wood",
        "Stormwater management systems prevent flooding in densely developed urban areas",
        "The historic preservation ordinance protects buildings older than seventy-five years",
        "Sustainable design principles include passive solar heating and natural ventilation",
        "The zoning variance allows the building to exceed the standard height limit",
        "Accessible design ensures buildings can be used by people of all physical abilities",
        "The landscape architect specified native drought-resistant plants for the public plaza",
        "Transit-oriented development concentrates housing and services near train stations",
        "The structural engineer calculated the maximum load capacity of the floor system",
        "Infill development uses vacant or underused parcels within existing urban areas",
        "The building envelope must meet thermal performance standards for energy efficiency",
        "Community engagement sessions gathered resident input on the neighborhood redesign",
        "The parking ratio determines how many spaces are required per unit of building area",
        "Modular construction assembles factory-built components on site to reduce build time",
    ])

    # ---- Retail / customer experience ----
    sentences.extend([
        "The loyalty program rewards customers with points for every dollar spent",
        "The checkout process should be streamlined to minimize cart abandonment rates",
        "Visual merchandising arranges products to attract attention and encourage purchases",
        "The customer satisfaction score improved after redesigning the returns process",
        "Omnichannel retailing provides a seamless experience across online and in-store shopping",
        "The pop-up shop will operate for three months in the vacant downtown storefront",
        "Inventory replenishment happens automatically when stock levels fall below thresholds",
        "The store layout guides customers through high-margin departments before essentials",
        "Mystery shoppers evaluate service quality by posing as regular customers",
        "The point-of-sale system processes payments and tracks sales data in real time",
        "Seasonal promotions drive traffic during traditionally slower shopping periods",
        "The personal shopping service helps customers find gifts and coordinate outfits",
        "Same-day delivery options have become a competitive differentiator for retailers",
        "The customer complaint was resolved with a full refund and a discount coupon",
        "Product placement at eye level on shelves increases likelihood of purchase",
        "The self-checkout kiosks reduce wait times during peak shopping hours",
        "The gift registry allows guests to purchase items from a curated wish list",
        "Price matching policies guarantee customers receive the lowest available price",
        "The flagship store serves as the brand's showcase and primary customer touchpoint",
        "Subscription box services deliver curated products to customers on a recurring basis",
    ])

    # ---- Linguistics / language ----
    sentences.extend([
        "Syntax rules determine the structure and order of words in a sentence",
        "Morphemes are the smallest meaningful units of language",
        "Bilingual children often switch between languages within a single conversation",
        "The phonetic alphabet provides a standardized way to transcribe speech sounds",
        "Language acquisition in children follows predictable stages of development",
        "Dialects reflect regional and social variations within the same language",
        "The translator must capture not just meaning but also tone and cultural context",
        "Sign language is a complete linguistic system with its own grammar and syntax",
        "Etymology traces the historical origin and evolution of individual words",
        "Pragmatics studies how context influences the interpretation of spoken language",
        "The endangered language has fewer than a hundred remaining native speakers",
        "Loanwords are terms adopted from one language into another through contact",
        "The speech pathologist works with children who have articulation difficulties",
        "Creole languages develop from contact between speakers of different native tongues",
        "Computational linguistics develops algorithms for machine translation systems",
        "The dictionary includes pronunciation guides using the international phonetic alphabet",
        "Semantic ambiguity occurs when a word or phrase has multiple possible meanings",
        "Pidgin languages emerge as simplified communication systems between groups",
        "The writing system uses logographic characters rather than an alphabet",
        "Discourse analysis examines how language functions in social and cultural contexts",
    ])

    # ---- Photography / visual media ----
    sentences.extend([
        "The portrait photographer uses natural window light for a soft flattering effect",
        "Aperture settings control the depth of field and amount of background blur",
        "The time-lapse video shows clouds moving across the sky over several hours",
        "Color correction adjusts the white balance and exposure of the raw image file",
        "The wildlife photographer waited three days for the eagle to return to its nest",
        "Composition techniques include the rule of thirds leading lines and framing",
        "The drone captured aerial footage of the coastline during golden hour",
        "Post-processing software allows detailed editing of contrast saturation and sharpness",
        "The macro lens reveals intricate details invisible to the unaided eye",
        "Street photography captures candid moments of everyday urban life",
        "The studio lighting setup includes a key light fill light and backlight",
        "Long exposure photography creates smooth flowing effects in moving water",
        "The photo editing workflow includes culling selecting and batch processing images",
        "High dynamic range imaging combines multiple exposures for balanced lighting",
        "The documentary photographer spent a year embedded with the community",
        "Mirrorless cameras offer the image quality of DSLRs in a more compact body",
        "The gallery prints are produced on archival paper to prevent fading over time",
        "Focus stacking combines multiple images to extend the depth of sharp focus",
        "The event photographer delivered edited images within forty-eight hours",
        "Black and white conversion emphasizes texture contrast and emotional mood",
    ])

    # ---- Geology / earth science ----
    sentences.extend([
        "Sedimentary rocks form through the accumulation and compression of deposits",
        "The geologist collected core samples to study underground rock formations",
        "Earthquakes occur along fault lines where tectonic plates interact and shift",
        "Mineral identification involves testing hardness luster and crystal structure",
        "Erosion by wind and water gradually reshapes the landscape over millennia",
        "The volcanic eruption deposited a thick layer of ash across the surrounding valley",
        "Groundwater flows through permeable rock layers known as aquifers",
        "Metamorphic rocks form when existing rocks are altered by heat and pressure",
        "The geological survey mapped subsurface features using seismic reflection data",
        "Fossilized remains in sedimentary layers provide evidence of ancient life forms",
        "Plate tectonics explains the movement and interaction of the earth's crustal plates",
        "The karst landscape features sinkholes caves and underground drainage systems",
        "Radioactive isotope dating determines the absolute age of rock specimens",
        "The mineral deposits in the region make it a significant source of copper and zinc",
        "Glacial moraines are ridges of debris deposited by retreating ice sheets",
        "The soil profile shows distinct layers formed by weathering and biological activity",
        "Geothermal springs indicate the presence of heated groundwater near volcanic zones",
        "The landslide was triggered by heavy rainfall saturating the unstable hillside",
        "Continental drift theory explains how landmasses have changed position over time",
        "The petrographic microscope reveals the mineral composition of thin rock sections",
    ])

    # ---- Emergency services / public safety ----
    sentences.extend([
        "The dispatcher received the emergency call and sent the nearest available unit",
        "Firefighters train extensively in structural collapse rescue techniques",
        "The evacuation plan designates assembly points for all building occupants",
        "Paramedics administered first aid at the scene before transporting to the hospital",
        "The severe weather warning system alerts residents through sirens and mobile devices",
        "Search and rescue teams use trained dogs to locate missing persons in wilderness",
        "The fire suppression system activates automatically when smoke is detected",
        "Community emergency response teams assist professional responders during disasters",
        "The hazardous materials team contains and cleans up chemical spills safely",
        "Public safety campaigns promote awareness of fire prevention and escape planning",
        "The emergency operations center coordinates multi-agency response efforts",
        "Active shooter training teaches employees the run hide fight response protocol",
        "The ambulance service response time averages under eight minutes in urban areas",
        "Building fire codes require sprinkler systems in commercial structures above a certain size",
        "The water rescue team deployed inflatable boats during the flash flood event",
        "Automated external defibrillators are placed in public buildings for cardiac emergencies",
        "The incident command system establishes a clear chain of authority at the scene",
        "Wildland firefighters create containment lines to prevent fire from spreading further",
        "The crisis negotiation team works to resolve standoff situations peacefully",
        "Emergency preparedness kits should include water food medications and flashlights",
    ])

    # ---- Marketing / advertising ----
    sentences.extend([
        "The social media campaign reached three million impressions in the first week",
        "Brand positioning differentiates the product from competitors in the consumer's mind",
        "The email newsletter has an open rate of twenty-two percent across all segments",
        "Search engine optimization improves the visibility of web pages in organic results",
        "The advertising agency developed a thirty-second television commercial for the launch",
        "Content marketing attracts potential customers through valuable informative material",
        "The conversion rate measures the percentage of visitors who complete a desired action",
        "Influencer partnerships amplify brand awareness among targeted demographic groups",
        "The market research survey gathered preferences from a sample of five hundred consumers",
        "Pay-per-click advertising charges the advertiser each time a user clicks the ad",
        "The brand identity guidelines specify colors fonts and logo usage rules",
        "A/B testing compares two versions of a webpage to determine which performs better",
        "The product launch event generated significant media coverage and social buzz",
        "Customer segmentation divides the audience into groups based on shared characteristics",
        "The banner ad placement on high-traffic websites drives awareness for new products",
        "Retargeting campaigns show ads to users who previously visited the website",
        "The public relations team drafted a press release announcing the company milestone",
        "Geographic targeting ensures advertisements are shown only to users in specific regions",
        "The marketing funnel maps the customer journey from awareness to purchase decision",
        "Affiliate marketing programs pay commissions for sales generated through partner links",
    ])

    # ---- Childcare / early development ----
    sentences.extend([
        "Sensory play activities help toddlers explore textures sounds and colors",
        "The preschool curriculum includes circle time outdoor play and creative arts",
        "Fine motor skills develop as children practice cutting drawing and building with blocks",
        "The speech therapist works with children who have delayed language development",
        "Imaginative play encourages children to develop problem-solving and social skills",
        "The infant feeding schedule transitions from formula to solid foods around six months",
        "Learning through music helps children develop rhythm language and coordination",
        "The childcare facility maintains a ratio of one adult for every four toddlers",
        "Picture books with repetitive text help early readers build confidence and vocabulary",
        "Outdoor nature walks expose children to science concepts through direct observation",
        "The occupational therapist helps children develop skills for self-care and school tasks",
        "Water table play teaches children about volume pouring and cause and effect",
        "The developmental screening identifies children who may benefit from early intervention",
        "Structured routines give young children a sense of security and predictability",
        "Cooperative games teach children to work together rather than compete against each other",
    ])

    # ---- Additional tech support / troubleshooting ----
    sentences.extend([
        "The graphics card overheats and causes the display to freeze during gaming sessions",
        "Clear the browser cache and cookies to resolve persistent login issues",
        "The network administrator configured the firewall to block unauthorized access attempts",
        "Check the event viewer logs for error messages related to the application crash",
        "The RAID array rebuild process can take several hours depending on disk capacity",
        "Ensure the BIOS firmware is updated to the latest version for hardware compatibility",
        "The SSL certificate expired and visitors see a security warning on the website",
        "Defragmenting the hard drive can improve file access speed on older systems",
        "The email server quota has been reached and new messages are being bounced back",
        "Port forwarding allows external devices to access services on the local network",
        "The virtual machine requires at least eight gigabytes of RAM to run smoothly",
        "Updating the device drivers resolved the peripheral compatibility issue",
        "The DNS server is not responding which prevents websites from loading",
        "Factory resetting the router restores default settings and often fixes connectivity",
        "The backup power supply activates when the main power source is interrupted",
    ])

    # ---- Additional product reviews / opinions ----
    sentences.extend([
        "The noise cancelling feature works remarkably well on airplane flights",
        "Battery life is impressive and easily lasts the entire workday on a single charge",
        "The build quality feels premium with a solid aluminum unibody construction",
        "Setup was straightforward and I was up and running in under ten minutes",
        "The camera quality exceeds expectations especially in low-light conditions",
        "Customer support was responsive and replaced the defective unit within days",
        "The user interface is intuitive and requires almost no learning curve",
        "Sound quality is rich and balanced with clear highs and deep bass response",
        "The product arrived well-packaged with no signs of damage during shipping",
        "The software updates have steadily improved performance since the initial release",
        "Portability is a major advantage as the device weighs less than two pounds",
        "The screen resolution is sharp enough to read small text comfortably",
        "Value for money is excellent considering the features included at this price point",
        "The warranty service was hassle-free and the repair was completed promptly",
        "Compatibility with existing accessories and peripherals works without any issues",
    ])

    # ---- Additional diverse cross-domain sentences ----
    sentences.extend([
        "The forensic accountant traced the fraudulent transactions through multiple shell companies",
        "Automated greenhouse systems control temperature humidity and watering schedules",
        "The translation service supports over forty languages including rare dialect variants",
        "Dental implants provide a permanent replacement for missing teeth with natural appearance",
        "The microbrewery produces small batches of handcrafted ale using traditional methods",
        "Seismic retrofitting strengthens older buildings to withstand earthquake forces",
        "The book club meets monthly to discuss contemporary and classic literature selections",
        "Precision medicine tailors treatment plans based on individual genetic profiles",
        "The composting facility processes municipal organic waste into commercial-grade fertilizer",
        "Augmented reality navigation overlays directional arrows onto the live camera view",
        "The scholarship committee reviewed applications from over three hundred eligible students",
        "Geospatial analysis combines satellite imagery with ground-level data for mapping",
        "The meditation retreat center offers week-long programs for stress reduction and renewal",
        "Hydraulic fracturing extracts natural gas from deep shale rock formations underground",
        "The outdoor amphitheater hosts summer concerts and theatrical performances for the public",
        "Cryptographic protocols secure online banking transactions from interception and tampering",
        "The veterinary clinic offers emergency care around the clock for critically ill animals",
        "Industrial water recycling systems reduce freshwater consumption in manufacturing plants",
        "The film restoration project preserved classic movies using digital scanning technology",
        "Drone inspection services examine power lines and cell towers without scaffolding",
        "The community land trust provides permanently affordable housing through shared ownership",
        "Proteomics research studies the complete set of proteins expressed by a cell or organism",
        "The antique appraisal service provides written valuations for estate and insurance purposes",
        "Noise-cancelling technology uses inverse sound waves to reduce ambient background noise",
        "The public library system offers free access to digital audiobooks and streaming services",
        "Biomass energy converts plant material and agricultural waste into renewable fuel",
        "The cycling advocacy group lobbies for protected bike lanes on major city streets",
        "Spectroscopy identifies chemical compounds based on their interaction with light energy",
        "The catering company provides customized menus for weddings corporate events and galas",
        "Satellite radio delivers uninterrupted music and talk programming across the continent",
        "The consumer protection agency investigates complaints about misleading product claims",
        "Permaculture design creates self-sustaining agricultural ecosystems modeled on nature",
        "The dental hygienist performs cleanings and screens for signs of oral disease",
        "Wearable fitness technology tracks steps heart rate and sleep quality throughout the day",
        "The youth mentoring program matches adult volunteers with at-risk teenagers",
        "Plasma cutting uses a high-velocity jet of ionized gas to slice through metal",
        "The subscription management platform handles billing renewals and cancellations",
        "Archaeological dating methods include radiocarbon analysis and stratigraphic comparison",
        "The community orchestra welcomes musicians of all skill levels to rehearse and perform",
        "Phytoremediation uses specific plant species to extract pollutants from contaminated soil",
        "The rare book dealer specializes in first editions and signed literary manuscripts",
        "Industrial robotics perform repetitive tasks with greater speed and consistency than humans",
        "The farmers market accepts electronic payment through a mobile card reader device",
        "Network security monitoring detects unusual traffic patterns that may indicate intrusion",
        "The heritage railway operates vintage steam locomotives on scenic mountain routes",
        "Prosthetic limb technology has advanced to include sensor-controlled articulating joints",
        "The animal control officer responds to reports of stray and potentially dangerous animals",
        "Photovoltaic efficiency has improved steadily as solar cell materials have advanced",
        "The escape room business designs themed puzzle challenges for groups of all ages",
        "Cloud-based document collaboration allows multiple users to edit files simultaneously",
        "The botanical garden maintains collections of rare and endangered plant species",
        "Robotic surgery systems enable minimally invasive procedures with enhanced precision",
        "The food co-op distributes weekly shares of locally grown seasonal produce to members",
        "Digital twin technology simulates building performance before construction begins",
        "The after-school robotics club competes in regional engineering design challenges",
        "Bioacoustics research records and analyzes the sounds produced by wildlife species",
        "The certified public accountant prepares tax returns for individuals and small businesses",
        "Vertical takeoff aircraft technology enables air taxi services in congested urban areas",
        "The home energy storage battery charges from solar panels during daylight hours",
        "Artifact conservation requires careful cleaning stabilization and climate-controlled storage",
        "The teletherapy platform connects patients with licensed counselors via secure video",
        "Regenerative agriculture practices build soil health while sequestering atmospheric carbon",
        "The mobile blood donation unit visits corporate offices and community centers weekly",
        "Digital currency exchanges facilitate the buying and selling of cryptocurrencies",
        "The conflict resolution mediator helps disputing parties reach mutually acceptable agreements",
        "Precision fermentation produces animal-free dairy proteins using engineered microorganisms",
        "The ice rink provides public skating sessions and hockey league games throughout winter",
        "Electromagnetic pulse shielding protects sensitive electronics from power surge damage",
        "The adaptive reuse project converted the former warehouse into residential loft apartments",
        "Citizen science projects invite the public to contribute data for ongoing research studies",
        "The wine sommelier recommends pairings that complement each course of the tasting menu",
        "Ocean wave energy converters transform the motion of waves into usable electricity",
        "The adult literacy program provides free evening classes for community members",
        "Biochar soil amendment improves water retention and nutrient availability for crops",
        "The coworking space provides flexible desk rentals meeting rooms and networking events",
        "Nanoparticle drug delivery systems target specific cells to improve treatment effectiveness",
        "The parkour gym teaches movement skills including vaulting climbing and precision landing",
        "Blockchain-based supply chain tracking verifies the origin and handling of food products",
        "The historical society archives photographs documents and oral histories from the region",
        "Algae cultivation produces biomass that can be converted into biofuel or animal feed",
        "The petting zoo introduces children to farm animals in a safe supervised environment",
        "Haptic feedback technology simulates the sense of touch in virtual reality environments",
        "The conflict-free diamond certification ensures stones are sourced from ethical mines",
        "Vertical axis wind turbines operate effectively in turbulent urban wind conditions",
        "The clinical trial enrolled patients to evaluate the safety and efficacy of the new drug",
        "Smart irrigation controllers adjust watering schedules based on weather forecast data",
    ])

    # ---- Hospitality / food service ----
    sentences.extend([
        "The restaurant manager trained new staff on food safety handling procedures",
        "Table reservation software manages seating capacity and wait list notifications",
        "The wine list features selections from regional vineyards and international imports",
        "Kitchen inventory management minimizes food waste and controls ingredient costs",
        "The banquet hall accommodates up to two hundred guests for formal events",
        "Menu engineering analyzes profitability and popularity of each dish offered",
        "The health inspector grades restaurants on sanitation food storage and preparation",
        "Front-of-house staff are trained in customer service and conflict de-escalation",
        "The catering order includes dietary accommodations for vegan and gluten-free guests",
        "Point-of-sale analytics reveal which menu items sell best during each shift",
        "The barista creates latte art using steamed milk and precise pouring technique",
        "Restaurant profit margins are heavily influenced by food cost and labor expenses",
        "The chef's tasting menu features seven courses with wine pairings for each",
        "Online ordering platforms charge restaurants a commission on each delivered meal",
        "The dishwashing station uses a commercial high-temperature sanitizing machine",
        "Seasonal menu changes highlight ingredients that are freshly available locally",
        "The host stand manages walk-in guests and coordinates with the kitchen on timing",
        "Food truck permits require compliance with local health and zoning regulations",
        "The cocktail menu includes signature drinks crafted with house-made syrups and bitters",
        "Staff scheduling software ensures adequate coverage during peak and off-peak hours",
    ])

    # ---- Cybersecurity / information security ----
    sentences.extend([
        "Multi-factor authentication adds an extra layer of security beyond just a password",
        "The penetration test identified critical vulnerabilities in the web application",
        "Ransomware attacks encrypt victim data and demand payment for the decryption key",
        "Security awareness training reduces the likelihood of employees falling for phishing",
        "The intrusion detection system monitors network traffic for suspicious activity",
        "Data loss prevention software blocks unauthorized transmission of sensitive files",
        "The security operations center monitors threats around the clock in shifts",
        "Patch management ensures all software is updated to address known vulnerabilities",
        "The incident response plan defines roles and procedures for handling data breaches",
        "Endpoint protection software defends individual devices against malware infections",
        "The access control list specifies which users can view or modify each resource",
        "Vulnerability scanning identifies weaknesses in systems before attackers can exploit them",
        "The security audit reviewed access logs and found several unauthorized login attempts",
        "Phishing emails impersonate trusted organizations to steal login credentials",
        "The encryption standard ensures data remains unreadable without the proper key",
        "Privilege escalation attacks gain higher access rights than originally authorized",
        "The firewall rules were updated to block traffic from known malicious IP addresses",
        "Social engineering attacks manipulate people into revealing confidential information",
        "The certificate authority verifies the identity of websites for secure connections",
        "Threat intelligence feeds provide real-time information about emerging attack methods",
    ])

    # ---- Sports commentary / game descriptions ----
    sentences.extend([
        "The quarterback threw a perfect spiral for a forty-yard touchdown pass",
        "The pitcher struck out the side to end the inning with the bases loaded",
        "The forward dribbled past three defenders and scored from outside the box",
        "The relay team set a new national record in the four-by-one-hundred meters",
        "The point guard dished a no-look pass for an easy layup under the basket",
        "The goalkeeper made a spectacular diving save to keep the score level",
        "The sprinter crossed the finish line just two hundredths of a second ahead",
        "The doubles team won the match in straight sets with powerful net play",
        "The running back broke through the defensive line for a fifteen-yard gain",
        "The gymnast executed a flawless routine on the balance beam for the top score",
        "The center fielder made a leaping catch at the warning track to rob a home run",
        "The power forward grabbed the offensive rebound and scored on the put-back",
        "The figure skater landed a triple axel for the first time in competition",
        "The midfielder delivered a cross that found the striker at the far post",
        "The wrestler pinned the opponent in the third period to win the championship",
    ])

    # ---- Additional e-commerce / shopping ----
    sentences.extend([
        "Free shipping is available on orders over fifty dollars within the continental US",
        "The product page includes customer photos alongside the professional images",
        "Size charts help customers choose the correct fit before placing an order",
        "The wishlist feature lets shoppers save items for future purchase consideration",
        "Estimated delivery time is three to five business days for standard shipping",
        "The product comparison tool displays features side by side for easy evaluation",
        "Gift wrapping is available for an additional charge during the checkout process",
        "The clearance section offers previous season items at significantly reduced prices",
        "Customer reviews mention that the product runs small and recommend sizing up",
        "The bundle deal includes the main product plus two accessories at a discounted price",
        "Back-in-stock notifications alert customers when sold-out items become available again",
        "The digital gift card can be sent instantly via email to the recipient",
        "Curbside pickup allows customers to order online and collect without entering the store",
        "The returns window extends to sixty days during the holiday shopping season",
        "Express checkout uses saved payment and shipping information for faster purchases",
    ])

    # ---- Additional medical / clinical ----
    sentences.extend([
        "The cardiologist ordered an echocardiogram to evaluate heart valve function",
        "Chronic obstructive pulmonary disease makes breathing progressively more difficult",
        "The pharmacist reviewed the prescription for potential drug interactions",
        "Laparoscopic surgery uses small incisions and a camera for minimally invasive procedures",
        "The oncologist discussed treatment options including chemotherapy and radiation therapy",
        "Blood type compatibility must be confirmed before performing a transfusion",
        "The dermatologist examined the mole and recommended a biopsy for further evaluation",
        "Physical rehabilitation after a stroke focuses on restoring motor function and speech",
        "The radiologist interpreted the CT scan and identified a small mass in the abdomen",
        "Telemedicine appointments allow patients in rural areas to consult with specialists",
        "The anesthesiologist monitors vital signs throughout the duration of the surgery",
        "Childhood vaccination schedules are designed to provide immunity at the earliest safe age",
        "The endocrinologist adjusted the insulin dosage based on blood sugar monitoring data",
        "Palliative care focuses on comfort and quality of life for terminally ill patients",
        "The orthopedic surgeon repaired the torn ligament using arthroscopic techniques",
    ])

    # ---- Additional legal / compliance ----
    sentences.extend([
        "The discovery process requires both parties to share relevant documents and evidence",
        "Class action lawsuits allow a group of plaintiffs to bring a common claim together",
        "The attorney filed a motion to dismiss the case due to lack of jurisdiction",
        "Regulatory compliance officers ensure the company follows all applicable laws",
        "The deposition transcript recorded the witness testimony under oath",
        "Antitrust laws prevent companies from forming monopolies that harm competition",
        "The appeals court overturned the lower court ruling based on procedural error",
        "Data protection regulations require explicit consent before collecting personal information",
        "The whistleblower protection statute shields employees who report illegal activity",
        "Contract amendments must be agreed upon in writing by all parties involved",
        "The environmental compliance report documents the company's adherence to emissions limits",
        "Intellectual property litigation can involve patents trademarks or copyrights",
        "The notary public certified the signatures on the real estate transfer documents",
        "Fiduciary duty requires financial advisors to act in the best interest of clients",
        "The compliance training module covers anti-bribery and corruption prevention policies",
    ])

    # ---- Additional finance / investing ----
    sentences.extend([
        "Dollar-cost averaging reduces the impact of market volatility on investment purchases",
        "The index fund tracks the performance of the entire stock market at low cost",
        "Compound interest accelerates wealth accumulation over long investment horizons",
        "The municipal bond provides tax-exempt interest income for eligible investors",
        "Asset allocation determines the proportion invested in stocks bonds and alternatives",
        "The options contract gives the holder the right but not obligation to buy shares",
        "Expense ratios measure the annual cost of managing an investment fund",
        "Rebalancing the portfolio restores the target allocation after market movements",
        "The earnings per share metric indicates how much profit each stock unit generates",
        "Tax-loss harvesting offsets capital gains by selling investments at a loss",
        "The price-to-earnings ratio compares share price to the company's annual profit",
        "Systematic investment plans automate regular contributions to retirement accounts",
        "Inflation-protected securities adjust their principal value based on consumer prices",
        "The dividend yield shows the annual dividend payment as a percentage of share price",
        "Target-date retirement funds automatically adjust risk allocation as the date approaches",
    ])

    # ---- Additional environment / climate ----
    sentences.extend([
        "Permafrost thawing releases stored methane which accelerates atmospheric warming",
        "The coral bleaching event was triggered by ocean temperatures above seasonal averages",
        "Agroforestry combines tree cultivation with crop production on the same land",
        "The carbon capture facility removes dioxide directly from industrial exhaust streams",
        "Microplastics have been detected in water sources across every continent",
        "The green roof installation reduces building heat absorption and manages stormwater runoff",
        "Marine protected areas restrict human activity to allow ecosystem recovery",
        "The lifecycle assessment measures environmental impact from production through disposal",
        "Pollinator decline threatens the production of crops that depend on insect fertilization",
        "The brownfield redevelopment project cleaned contaminated industrial land for new housing",
        "Circular economy principles design waste out of the production and consumption cycle",
        "The emissions trading system allows companies to buy and sell pollution allowances",
        "Biodegradable packaging breaks down naturally without leaving persistent residues",
        "The rewilding project reintroduced native species to restore the degraded ecosystem",
        "Environmental justice addresses the disproportionate pollution burden on vulnerable communities",
    ])

    # ---- Additional education / learning ----
    sentences.extend([
        "The flipped classroom model assigns lectures as homework and uses class time for practice",
        "Differentiated instruction adapts teaching methods to individual student learning styles",
        "The learning management system tracks student progress and assignment completion",
        "Project-based learning engages students in solving real-world problems over extended periods",
        "The reading intervention program provides additional support for struggling readers",
        "Formative assessment gives teachers ongoing feedback about student understanding",
        "The STEM curriculum integrates science technology engineering and mathematics concepts",
        "Universal design for learning provides multiple means of engagement and expression",
        "The gifted education program offers accelerated coursework and enrichment activities",
        "Social-emotional learning teaches students skills for managing emotions and relationships",
        "The special education team develops individualized education plans for qualifying students",
        "Gamification applies game design elements to educational activities for increased engagement",
        "The school counselor provides academic guidance and social-emotional support services",
        "Collaborative learning groups encourage students to discuss and solve problems together",
        "The digital citizenship curriculum teaches responsible and safe use of technology",
    ])

    # ---- Additional paraphrases (training W on semantic equivalence) ----
    sentences.extend([
        # Pair 36
        "The package was delivered to the wrong address by mistake",
        "The parcel was sent to an incorrect location in error",
        # Pair 37
        "The air conditioning unit is not cooling the room effectively",
        "The climate control system fails to lower the temperature adequately",
        # Pair 38
        "Please provide an estimated date for when the parts will arrive",
        "Can you give me an approximate timeline for receiving the components",
        # Pair 39
        "The vehicle was damaged during transport to the dealership",
        "The car sustained harm while being shipped to the sales location",
        # Pair 40
        "Annual revenue exceeded projections by twelve percent this year",
        "Yearly income surpassed forecasts by twelve percent in the current period",
        # Pair 41
        "The research paper cites over forty published academic sources",
        "The scholarly article references more than forty peer-reviewed publications",
        # Pair 42
        "Heavy snowfall blocked roads and caused widespread travel disruptions",
        "Significant winter precipitation closed highways and interrupted transportation",
        # Pair 43
        "The museum exhibit traces the evolution of communication technology",
        "The gallery display follows the development of messaging devices over time",
        # Pair 44
        "The landlord increased the monthly rent by five percent this year",
        "The property owner raised the periodic lease payment by five percent recently",
        # Pair 45
        "Volunteers cleaned up debris along the riverbank over the weekend",
        "Community helpers removed trash from the stream shore during Saturday and Sunday",
        # Pair 46
        "The project deadline was moved forward by two weeks unexpectedly",
        "The assignment due date was accelerated by a fortnight without prior warning",
        # Pair 47
        "Customer complaints about long wait times have increased this quarter",
        "Client grievances regarding extended hold durations rose during this period",
        # Pair 48
        "The surgeon performed the operation using a minimally invasive technique",
        "The doctor carried out the procedure with a less intrusive surgical method",
        # Pair 49
        "The new bridge connects two communities previously separated by the river",
        "The recently built span links two neighborhoods that the waterway had divided",
        # Pair 50
        "The board approved a plan to expand operations into three new countries",
        "The directors endorsed a strategy to grow the business across three additional nations",
        # Pair 51
        "The angry customer demanded a full refund immediately",
        "The irate client insisted on getting all their money back right away",
        # Pair 52
        "The software crashed and I lost all my unsaved work",
        "The application froze and all my unsaved progress was destroyed",
        # Pair 53
        "The hotel room was dirty and the sheets were stained",
        "The accommodation was unclean and the bed linen had marks on it",
        # Pair 54
        "Please confirm the appointment scheduled for tomorrow morning",
        "Can you verify the meeting that is set for tomorrow at the start of the day",
        # Pair 55
        "The test results came back negative for any serious condition",
        "The diagnostic findings showed no evidence of a severe illness",
        # Pair 56
        "The company hired fifty new employees to handle increased demand",
        "The business recruited fifty additional workers to manage the higher workload",
        # Pair 57
        "Internet speed is too slow for video streaming",
        "The network connection is insufficient for watching online content",
        # Pair 58
        "The restaurant received a poor hygiene rating from inspectors",
        "The dining establishment was given a low cleanliness score by health officials",
        # Pair 59
        "My order was incomplete and several items were missing",
        "The shipment I received was partial and lacked multiple products",
        # Pair 60
        "The teacher praised the student for outstanding academic performance",
        "The instructor commended the pupil for exceptional scholarly achievement",
        # Pair 61
        "The plumber fixed the burst pipe in the basement",
        "The tradesperson repaired the broken water line in the lower level",
        # Pair 62
        "Global temperatures have risen by one degree over the past century",
        "Worldwide heat levels increased by one degree during the last hundred years",
        # Pair 63
        "The baby has been crying for hours and won't stop",
        "The infant has been wailing for a long time and refuses to be calmed",
        # Pair 64
        "The CEO resigned amid allegations of financial misconduct",
        "The chief executive stepped down following claims of monetary wrongdoing",
        # Pair 65
        "The garden needs watering because it hasn't rained in weeks",
        "The yard requires irrigation since there has been no precipitation recently",
        # Pair 66
        "Customers are complaining about the long wait times on the phone",
        "Clients are expressing frustration about extended hold periods when calling",
        # Pair 67
        "The athlete injured her knee during the championship game",
        "The sports player hurt her knee joint in the title match",
        # Pair 68
        "The bank approved the mortgage application after reviewing documents",
        "The financial institution accepted the home loan request following a paperwork review",
        # Pair 69
        "Traffic was diverted because of a major accident on the highway",
        "Vehicles were rerouted due to a serious collision on the motorway",
        # Pair 70
        "The museum is closed on Mondays for routine maintenance",
        "The gallery does not operate on the first day of the week for regular upkeep",
        # Pair 71
        "She wrote a lengthy complaint about the poor customer service",
        "She drafted an extensive grievance regarding the inadequate client support",
        # Pair 72
        "The plane was delayed by three hours due to bad weather",
        "The aircraft departure was postponed for three hours because of adverse conditions",
        # Pair 73
        "The book became a bestseller within the first month of publication",
        "The novel reached the top of sales charts within weeks of being released",
        # Pair 74
        "The tenant reported a broken window in the apartment",
        "The renter notified management about a damaged pane in the flat",
        # Pair 75
        "The company recalled thousands of defective products from stores",
        "The manufacturer pulled back thousands of faulty items from retail outlets",
        # Pair 76
        "The child was awarded first prize in the science competition",
        "The young student won the top award at the scientific contest",
        # Pair 77
        "The electricity went out during the thunderstorm last night",
        "The power failed during the severe lightning storm yesterday evening",
        # Pair 78
        "He struggled to assemble the furniture without clear instructions",
        "He had difficulty putting together the furnishings due to vague directions",
        # Pair 79
        "The restaurant offers a discount for seniors on weekday afternoons",
        "The eatery provides a reduced price for elderly patrons during weekday lunch hours",
        # Pair 80
        "The doctor prescribed antibiotics for the bacterial infection",
        "The physician ordered antimicrobial medication to treat the bacterial illness",
        # Pair 81
        "Rent prices in the city have skyrocketed over the past five years",
        "Lease costs in the urban area have surged dramatically in recent years",
        # Pair 82
        "The supervisor reprimanded the employee for arriving late repeatedly",
        "The manager disciplined the worker for habitual tardiness",
        # Pair 83
        "The charity event raised over ten thousand dollars for local families",
        "The fundraising occasion generated more than ten thousand dollars for nearby households",
        # Pair 84
        "The students protested against the tuition increase at the university",
        "The learners demonstrated in opposition to the fee hike at the college",
        # Pair 85
        "The delivery driver left the package at the wrong house",
        "The courier placed the parcel at an incorrect residence",
        # Pair 86
        "The new smartphone has an excellent camera with optical zoom",
        "The latest mobile phone features a superb lens with optical magnification",
        # Pair 87
        "The water heater broke and there is no hot water in the house",
        "The heating unit malfunctioned and warm water is unavailable in the home",
        # Pair 88
        "The government announced new regulations for data privacy",
        "The authorities introduced fresh rules governing information protection",
        # Pair 89
        "The toddler refuses to eat vegetables at every meal",
        "The small child will not consume greens at any mealtime",
        # Pair 90
        "The concert was cancelled due to the lead singer's illness",
        "The musical performance was called off because the primary vocalist was sick",
        # Pair 91
        "She is nervous about the upcoming job interview next week",
        "She feels anxious about the employment meeting happening in a few days",
        # Pair 92
        "The hurricane caused extensive damage to coastal properties",
        "The tropical cyclone inflicted severe harm on seaside buildings",
        # Pair 93
        "The mechanic said the car needs new brake pads urgently",
        "The auto technician indicated the vehicle requires replacement stopping pads immediately",
        # Pair 94
        "The professor cancelled the lecture because of a personal emergency",
        "The academic called off the class session due to a private urgent matter",
        # Pair 95
        "The website keeps showing error messages when I try to log in",
        "The web platform repeatedly displays fault notifications when I attempt to sign in",
        # Pair 96
        "Her presentation impressed the entire audience at the conference",
        "Her talk captivated all attendees at the professional gathering",
        # Pair 97
        "The coffee shop closes early on Sundays and holidays",
        "The cafe has reduced hours on the last day of the week and festive occasions",
        # Pair 98
        "The warehouse is running out of stock on several popular items",
        "The storage facility is depleting inventory on multiple high-demand products",
        # Pair 99
        "The patient experienced severe side effects from the medication",
        "The individual suffered intense adverse reactions to the prescribed drug",
        # Pair 100
        "The fire department responded quickly to the emergency call",
        "The firefighting unit arrived rapidly after the distress notification",
        # Pair 101
        "She returned the dress because it did not fit properly",
        "She sent back the garment since it was the wrong size",
        # Pair 102
        "The construction project is behind schedule due to supply shortages",
        "The building work is delayed because of material scarcity",
        # Pair 103
        "He apologized for the misunderstanding and offered to make it right",
        "He expressed regret for the confusion and proposed to correct the situation",
        # Pair 104
        "The network outage affected thousands of users across the region",
        "The connectivity failure impacted numerous subscribers throughout the area",
        # Pair 105
        "The real estate agent showed us five houses in one afternoon",
        "The property broker arranged viewings of five homes during a single afternoon session",
        # Pair 106
        "The new policy requires all employees to wear identification badges",
        "The updated rule mandates that every worker must display name tags",
        # Pair 107
        "The river overflowed its banks after days of continuous rainfall",
        "The waterway breached its boundaries following sustained precipitation",
        # Pair 108
        "The airline lost my luggage and hasn't located it yet",
        "The carrier misplaced my baggage and has not found it so far",
        # Pair 109
        "The teacher gave the class extra homework over the holiday break",
        "The instructor assigned additional schoolwork during the vacation period",
        # Pair 110
        "The city plans to build a new public park in the downtown area",
        "The municipality intends to construct a new recreational green space in the urban core",
        # Pair 111
        "I am extremely unhappy with how my complaint was handled",
        "I am very dissatisfied with the way my grievance was addressed",
        # Pair 112
        "The paint is peeling off the walls in several rooms",
        "The coating is flaking from the surfaces in multiple areas of the house",
        # Pair 113
        "The dog barks loudly every time a stranger approaches the door",
        "The canine makes loud noises whenever an unknown person comes near the entrance",
        # Pair 114
        "The quarterly sales figures were disappointing compared to last year",
        "The three-month revenue numbers were underwhelming relative to the prior year",
        # Pair 115
        "She found a bug in the code that caused incorrect calculations",
        "She discovered a defect in the program that produced wrong results",
        # Pair 116
        "The landlord refuses to repair the broken heating system",
        "The property owner will not fix the malfunctioning furnace",
        # Pair 117
        "The committee voted unanimously to approve the budget proposal",
        "The panel reached full agreement to accept the financial plan",
        # Pair 118
        "The new employee is struggling to learn the company's software systems",
        "The recent hire is having difficulty mastering the organization's digital tools",
        # Pair 119
        "The customer left a one-star review due to terrible service",
        "The patron posted the lowest rating because of awful support",
        # Pair 120
        "The bridge was closed for repairs after inspectors found structural cracks",
        "The overpass was shut down for maintenance after assessors discovered fractures in the framework",
        # Pair 121
        "I would like to speak with a manager about this issue",
        "I want to talk to someone in charge regarding this problem",
        # Pair 122
        "The surgeon successfully removed the tumor during the operation",
        "The doctor effectively excised the growth during the surgical procedure",
        # Pair 123
        "Gas prices have dropped significantly since last summer",
        "Fuel costs have decreased considerably compared to the previous warm season",
        # Pair 124
        "The printer is jammed and I cannot get the paper out",
        "The printing device is stuck and I am unable to remove the sheets",
        # Pair 125
        "The students organized a fundraiser to support the local animal shelter",
        "The pupils arranged a charity drive to aid the nearby pet rescue organization",
        # Pair 126
        "I was charged twice for the same purchase on my credit card",
        "My payment card was billed double for a single transaction",
        # Pair 127
        "The jury found the defendant guilty on all counts",
        "The panel of judges determined the accused was culpable on every charge",
        # Pair 128
        "The hiker got lost in the mountains and had to call for rescue",
        "The trekker became disoriented in the highlands and needed to summon help",
        # Pair 129
        "The company is expanding its product line to include organic options",
        "The business is broadening its offerings to encompass natural alternatives",
        # Pair 130
        "The roof leaks every time it rains heavily",
        "The top of the house lets water in during intense downpours",
        # Pair 131
        "The flight attendant was rude and dismissive to passengers",
        "The cabin crew member was impolite and disregarding toward travelers",
        # Pair 132
        "The school principal announced a new anti-bullying initiative",
        "The head of school revealed a fresh program to combat harassment among students",
        # Pair 133
        "The shipment arrived damaged and several items were broken",
        "The delivery came with harm to the contents and multiple products were shattered",
        # Pair 134
        "The dentist recommended getting braces to correct the alignment",
        "The oral care specialist suggested orthodontic treatment to fix the positioning",
        # Pair 135
        "The senator introduced a bill to increase funding for infrastructure",
        "The legislator proposed new legislation to boost spending on public works",
        # Pair 136
        "The washing machine is leaking water all over the laundry room floor",
        "The clothes cleaning appliance is dripping fluid across the utility room surface",
        # Pair 137
        "The startup went bankrupt after failing to secure additional funding",
        "The young company became insolvent after being unable to obtain more investment",
        # Pair 138
        "I need someone to fix my air conditioning before the heatwave",
        "I require a technician to repair my cooling system ahead of the extreme temperatures",
        # Pair 139
        "The teacher noticed the child was being bullied during recess",
        "The educator observed that the kid was being harassed at playtime",
        # Pair 140
        "She completed the marathon in under four hours despite the heat",
        "She finished the long-distance race in less than four hours even in the hot weather",
        # Pair 141
        "The insurance company denied my claim without a valid explanation",
        "The coverage provider rejected my request with no proper justification",
        # Pair 142
        "The city experienced a record-breaking snowfall last winter",
        "The metropolitan area had unprecedented snow accumulation during the previous cold season",
        # Pair 143
        "The intern made a significant contribution to the research project",
        "The trainee provided a meaningful input to the study initiative",
        # Pair 144
        "The neighbors are making excessive noise late at night",
        "The people next door are being extremely loud during nighttime hours",
        # Pair 145
        "The vet recommended a special diet for the overweight dog",
        "The animal doctor suggested a specific meal plan for the heavy canine",
        # Pair 146
        "The elevator has been out of service for over a week now",
        "The lift has been broken and non-functional for more than seven days",
        # Pair 147
        "The store manager apologized for the incorrect price on the shelf",
        "The shop supervisor expressed regret about the wrong label displayed for the item",
        # Pair 148
        "I was promised a callback but nobody ever contacted me",
        "They assured me someone would return my call but I never heard from anyone",
        # Pair 149
        "The thunderstorm knocked out power for the entire neighborhood",
        "The electrical storm caused a blackout across the whole residential area",
        # Pair 150
        "The librarian helped the student find reliable sources for the essay",
        "The library staff assisted the pupil in locating credible references for the paper",
    ])

    # ---- Additional paraphrase pairs for sentiment and complaint equivalences ----
    sentences.extend([
        # Complaint / frustration paraphrases
        "I am furious about the billing error on my account",
        "I am enraged by the charging mistake on my statement",
        "Your customer service is absolutely terrible",
        "The support team provides an appalling level of assistance",
        "I want to file a formal complaint about this experience",
        "I wish to submit an official grievance regarding this situation",
        "The product broke after only two days of use",
        "The item stopped working just two days after I started using it",
        "Nobody has responded to my emails for over a week",
        "My messages have gone unanswered for more than seven days",
        "I was treated unfairly by your staff at the store",
        "The employees at the shop were unjust in their treatment of me",
        "This is completely unacceptable and I demand a resolution",
        "The current situation cannot be tolerated and I insist on a fix",
        "The quality has gone downhill since I first subscribed",
        "Standards have deteriorated since I initially signed up",
        "I feel like my concerns are being ignored entirely",
        "It seems as though my worries are being completely disregarded",
        "The technician was unprofessional and arrived two hours late",
        "The service person lacked professionalism and showed up well past the scheduled time",
    ])

    # ---- Additional short query-style entries ----
    sentences.extend([
        "frustrated customer complaint",
        "angry about bad service",
        "demand refund defective product",
        "billing error overcharged",
        "terrible customer support",
        "product not working broken",
        "delivery never arrived lost",
        "wrong item shipped return",
        "rude staff behavior complaint",
        "long wait time phone support",
        "subscription cancellation problems",
        "unauthorized charge credit card",
        "damaged goods received",
        "missing items from order",
        "service quality declined",
        "unreliable product disappointed",
        "late delivery no update",
        "account locked cannot access",
        "poor communication no response",
        "faulty equipment replacement",
        "warranty claim denied unfair",
        "hidden fees unexpected charges",
        "false advertising misleading",
        "unsafe product safety concern",
        "noise complaint neighbor dispute",
        "landlord not fixing repairs",
        "insurance claim rejected",
        "medical billing mistake",
        "appointment cancelled no notice",
        "website down cannot order",
        "data breach security concern",
        "spam emails unwanted messages",
        "slow internet connection",
        "app keeps crashing freezing",
        "login issues password reset",
        "payment declined transaction failed",
        "contract dispute unfair terms",
        "food poisoning restaurant",
        "hotel room unclean dirty",
        "flight delay compensation",
        "car repair overcharged",
        "school complaint education issue",
        "noise pollution disturbing",
        "water leak plumbing emergency",
        "power outage electrical problem",
        "pest infestation exterminator",
        "mold problem health hazard",
        "parking violation ticket dispute",
        "tax return error correction",
        "travel booking cancellation refund",
        "gym membership termination",
        "job discrimination complaint",
        "harassment reporting procedure",
        "product recall safety alert",
        "environmental violation report",
        "building code violation",
        "traffic accident report",
        "stolen property theft report",
        "identity theft fraud alert",
        "child safety concern",
        "medication side effects adverse",
        "grade dispute academic appeal",
        "rent increase objection",
        "eviction notice dispute",
        "workplace safety violation",
        "pay discrepancy wage theft",
        "benefits denied appeal process",
        "accessibility complaint ADA",
        "privacy violation data misuse",
        "service outage downtime report",
        "emotional distress damages claim",
        "negligence liability lawsuit",
        "breach of contract remedies",
        "wrongful termination claim",
        "property damage repair estimate",
        "consumer rights protection",
        "unfair business practices",
        "defamation libel slander",
        "copyright infringement takedown",
        "patent violation dispute",
        "zoning violation appeal",
        "health code violation restaurant",
        "fire safety inspection failure",
        "utility billing dispute",
        "HOA dispute resolution",
        "neighbor boundary dispute",
        "tree removal permit",
        "construction noise complaint",
        "air quality concern pollution",
        "water quality test results",
        "road repair pothole report",
        "streetlight outage report",
        "public transit delay complaint",
        "school bus safety concern",
        "playground equipment broken",
        "park maintenance request",
        "library program suggestion",
        "community event planning",
        "volunteer coordination signup",
        "donation receipt tax deduction",
        "grant application deadline",
        "scholarship eligibility requirements",
    ])

    # ---- Feature request / product feedback paraphrases ----
    # These are critical: queries like "requesting new functionality" must map
    # to templates like "Can you add dark mode" or "We need bulk editing."
    sentences.extend([
        # Pair: requesting features
        "I would like to request a new feature for the product",
        "Can you please add new functionality to the platform",
        # Pair: suggesting improvements
        "Here is a feature suggestion for improving the dashboard",
        "I have an idea for enhancing the user interface",
        # Pair: wish list
        "This is on my wish list for future product updates",
        "I hope you will consider adding this in a future release",
        # Pair: product enhancement
        "We need a product enhancement for better reporting",
        "Please improve the reporting capabilities of the tool",
        # Pair: requesting new capabilities
        "Our team is requesting new capabilities for collaboration",
        "We need additional features to support team workflows",
        # Pair: feature gap
        "The product is missing a critical feature we need",
        "There is a gap in the functionality that must be addressed",
        # Pair: roadmap request
        "Is this feature on the product roadmap for this year",
        "When will this functionality be available in the product",
        # Pair: feedback and suggestions
        "I want to provide feedback and suggest improvements",
        "Let me share some ideas for making the product better",
        # Pair: dark mode as feature request
        "Can you add dark mode to the application",
        "We would like a dark theme option for the interface",
        # Pair: export as feature request
        "Please add CSV export to the analytics section",
        "We need the ability to download data as spreadsheet files",
        # Pair: bulk operations
        "We really need bulk editing capabilities",
        "Adding the ability to modify multiple records at once would be great",
        # Pair: mobile support
        "Would love to see a mobile app for this service",
        "Please make the platform accessible on smartphones and tablets",
        # Pair: permissions granularity
        "Can we get more granular role permissions",
        "We need finer access controls for different user types",
        # Pair: notifications
        "A notification system for alerts would save us time",
        "We need automated alerts when certain thresholds are reached",
        # Pair: audit trail
        "We need an audit trail feature for compliance",
        "Please add change tracking and logging for regulatory purposes",
        # Pair: scheduling
        "It would be great to schedule reports automatically",
        "We want recurring automated report generation capabilities",
    ])

    # ---- Short cross-domain phrases ----
    sentences.extend([
        "broken link on website",
        "package never arrived",
        "allergic reaction symptoms",
        "mortgage interest rate",
        "job interview preparation",
        "flight booking confirmation",
        "pet vaccination schedule",
        "garden pest control",
        "plumbing leak repair",
        "retirement savings calculator",
        "college admission requirements",
        "recipe substitution ideas",
        "car insurance quote",
        "yoga beginner routine",
        "apartment lease agreement",
        "credit score improvement",
        "home theater setup guide",
        "passport application status",
        "electric vehicle charging",
        "wedding venue availability",
        "vitamin supplement dosage",
        "concert ticket prices",
        "child custody arrangement",
        "roof repair estimate",
        "stock market analysis",
        "used car inspection checklist",
        "business loan application",
        "winter clothing essentials",
        "emergency contact information",
        "volunteer opportunity nearby",
    ])

    # ---- Pharmacy / medication ----
    sentences.extend([
        "The pharmacist verified the prescription dosage before dispensing the medication",
        "Generic medications contain the same active ingredients as brand-name equivalents",
        "Drug interactions can cause dangerous side effects when certain medications are combined",
        "The controlled substance prescription requires a valid government-issued identification",
        "Over-the-counter pain relievers include acetaminophen ibuprofen and aspirin options",
        "The medication guide explains potential side effects and proper storage conditions",
        "Compounding pharmacies prepare customized medication formulations for individual patients",
        "The automatic refill program sends prescription renewals without the patient requesting",
        "Antibiotic resistance develops when bacteria adapt to survive drug treatments",
        "The pharmacovigilance program tracks adverse drug reactions reported by healthcare providers",
        "Vaccines are stored at specific temperature ranges to maintain their effectiveness",
        "The drug formulary lists all medications covered under the insurance plan benefits",
        "Extended-release tablets deliver medication gradually over a longer period of time",
        "The pharmacy technician counted and labeled the prescription bottles for dispensing",
        "Clinical pharmacists provide medication therapy management during patient consultations",
    ])

    # ---- Social media / digital communication ----
    sentences.extend([
        "The viral post received millions of shares and comments within twenty-four hours",
        "Content moderation policies remove posts that violate community guidelines",
        "The algorithm prioritizes content based on user engagement and relevance signals",
        "Hashtag campaigns help organizations rally support around social causes",
        "The influencer's sponsored post disclosed the paid partnership as required by law",
        "Two-step verification protects social media accounts from unauthorized access",
        "The platform rolled out short-form video features to compete with rival services",
        "Online community managers respond to comments and foster positive discussions",
        "The privacy settings allow users to control who can view their profile information",
        "Misinformation spreads rapidly on social platforms without effective fact-checking",
        "The analytics dashboard shows follower growth engagement rates and post reach",
        "User-generated content provides authentic perspectives for brand marketing campaigns",
        "The reporting feature allows users to flag inappropriate or harmful content",
        "Live streaming enables real-time interaction between creators and their audiences",
        "The terms of service agreement outlines acceptable use of the social media platform",
    ])

    # ---- Construction / trades ----
    sentences.extend([
        "The concrete foundation must cure for at least seven days before construction continues",
        "The building inspector verified that framing meets structural code requirements",
        "Roofing materials include asphalt shingles metal panels and clay tiles",
        "The crane operator lifted steel beams into position for the upper floors",
        "Plumbing rough-in must be completed before the drywall installation begins",
        "The general contractor coordinates subcontractors and manages the project timeline",
        "Electrical wiring must follow the national electrical code for safety compliance",
        "The HVAC system design accounts for building size insulation and occupancy levels",
        "Concrete reinforced with steel rebar provides greater structural strength and durability",
        "The excavation crew dug the foundation trenches to the specified depth and width",
        "Window installation requires proper flashing and waterproof membrane application",
        "The site supervisor conducts daily safety briefings before work shifts begin",
        "Load calculations determine the size of structural beams and support columns",
        "The masonry crew laid brick in a running bond pattern for the exterior walls",
        "The project estimate includes materials labor equipment rental and permit fees",
    ])

    # ---- Aerospace / aviation ----
    sentences.extend([
        "The pilot performed a pre-flight inspection of all control surfaces and instruments",
        "Air traffic control manages the spacing and sequencing of aircraft near the airport",
        "The jet engine turbine converts fuel combustion into thrust for forward motion",
        "Aircraft maintenance logs document every inspection repair and part replacement",
        "The flight management computer calculates the most fuel-efficient route and altitude",
        "Composite materials reduce aircraft weight while maintaining structural strength",
        "The runway approach lighting system guides pilots during low-visibility landings",
        "Wind shear detection systems alert pilots to dangerous changes in wind speed",
        "The aircraft pressurization system maintains cabin altitude during high-altitude flight",
        "Unmanned aerial vehicles are used for surveillance mapping and agricultural spraying",
        "The avionics suite includes navigation communication and weather radar systems",
        "Noise abatement procedures minimize aircraft sound impact on residential areas",
        "The flight data recorder captures parameters used for accident investigation",
        "Deicing equipment removes ice accumulation from aircraft surfaces before takeoff",
        "The air traffic separation standard ensures safe distances between aircraft in flight",
    ])

    # ---- Nonprofit / volunteer ----
    sentences.extend([
        "The food bank distributed over fifty thousand meals during the holiday season",
        "The grant proposal outlines the project goals budget and expected outcomes",
        "Volunteer coordinators match available helpers with organization staffing needs",
        "The annual fundraising gala raised record donations for childhood education programs",
        "The donor database tracks giving history contact information and communication preferences",
        "The community outreach worker connects residents with available social services",
        "The nonprofit annual report discloses financial statements and program achievements",
        "Board governance training ensures directors understand their fiduciary responsibilities",
        "The thrift store revenue funds job training programs for underserved populations",
        "The volunteer appreciation event recognizes individuals who donated significant time",
        "The capital campaign raised funds to construct a new community recreation center",
        "Mission-driven organizations measure success by social impact rather than profit alone",
        "The advocacy coalition lobbies legislators to increase funding for mental health services",
        "In-kind donations of supplies and equipment supplement monetary contributions",
        "The pro bono legal clinic provides free consultations for low-income families",
    ])

    # ---- Data science / machine learning ----
    sentences.extend([
        "Feature engineering transforms raw data into variables that improve model predictions",
        "The confusion matrix shows true positives false positives and classification errors",
        "Cross-validation splits the dataset into folds to estimate model generalization",
        "The gradient descent algorithm iteratively adjusts parameters to minimize the loss function",
        "Random forests combine multiple decision trees to reduce overfitting and variance",
        "The ROC curve plots the tradeoff between true positive rate and false positive rate",
        "Dimensionality reduction techniques like PCA compress data while preserving variance",
        "The training pipeline includes data cleaning normalization and augmentation steps",
        "Hyperparameter tuning searches for the configuration that produces the best performance",
        "The embedding layer maps categorical variables into dense continuous vector representations",
        "Batch normalization stabilizes training by normalizing layer inputs during optimization",
        "The attention mechanism allows models to focus on relevant parts of the input sequence",
        "Regularization techniques prevent the model from memorizing noise in the training data",
        "The precision recall tradeoff depends on the relative cost of false positives and misses",
        "Ensemble methods combine predictions from multiple models to improve overall accuracy",
    ])

    # ---- Miscellaneous fill (diverse topics) ----
    sentences.extend([
        "The elevator inspection certificate must be displayed in the cab at all times",
        "Origami is the Japanese art of folding paper into decorative shapes without cutting",
        "The highway rest area provides restrooms picnic tables and vending machines",
        "Braille signage enables visually impaired individuals to navigate buildings independently",
        "The notary public witnessed the signing of the power of attorney document",
        "Tidal energy harnesses the predictable movement of ocean water for electricity generation",
        "The census questionnaire collects demographic data for government planning purposes",
        "Noise ordinances restrict loud activities during designated evening and nighttime hours",
        "The blood bank maintains an inventory of multiple blood types for hospital use",
        "Aeroponics suspends plant roots in air and delivers nutrients through a fine mist",
        "The time capsule was sealed and buried with instructions to open in fifty years",
        "Carbon dating estimates the age of organic materials based on radioactive decay rates",
        "The toll road charges vehicles automatically using electronic transponder technology",
        "Brainwave monitoring devices detect electrical activity patterns during sleep studies",
        "The flea market vendors sell antiques collectibles and handmade crafts every weekend",
        "Ultraviolet sterilization systems disinfect water by destroying harmful microorganisms",
        "The escape velocity required to leave Earth's gravitational pull is about eleven km per second",
        "The archives contain handwritten correspondence dating back to the eighteenth century",
        "Piezoelectric materials generate electricity when subjected to mechanical pressure",
        "The international date line runs through the Pacific Ocean and separates calendar days",
        "Bioluminescent organisms produce their own light through internal chemical reactions",
        "The puppet theater performs shows for children using hand-carved wooden characters",
        "Seismographs record ground vibrations to measure the magnitude and location of earthquakes",
        "The botanical illustrator draws detailed scientific diagrams of plant anatomy and structure",
        "Cryogenic preservation maintains biological samples at extremely low temperatures",
        "The street food vendor serves traditional dishes from a mobile cart on the sidewalk",
        "Cartography is the science and art of creating accurate maps and geographic charts",
        "The recycling facility sorts materials by type using optical sensors and conveyors",
        "Aquifer recharge projects inject treated water underground to replenish groundwater supplies",
        "The sundial uses the position of shadows cast by sunlight to indicate the time of day",
    ])

    # ---- Additional coverage: mixed domains for breadth ----
    sentences.extend([
        "The optometrist prescribed new lenses after the annual vision examination",
        "Geofencing technology triggers location-based alerts on mobile devices",
        "The wildlife corridor connects two protected habitats across the highway",
        "Supply chain transparency lets consumers trace products back to their origin",
        "The planetarium show explains the seasonal movement of constellations",
        "Occupational safety standards require harness use when working at heights",
        "The electric skateboard has a range of twenty miles on a single battery charge",
        "Inventory turnover ratio measures how quickly stock is sold and replaced",
        "The seed library allows gardeners to borrow and return heirloom seed varieties",
        "The autonomous delivery robot navigates sidewalks using cameras and sensors",
        "Noise-reducing pavement materials lower road traffic sound for nearby residents",
        "The beekeeping cooperative produces raw honey and beeswax candle products",
        "Remote patient monitoring devices transmit vital signs to the healthcare provider",
        "The underground cable network distributes electricity without overhead power lines",
        "Wave pool technology creates artificial surf conditions for indoor water parks",
        "The community radio station broadcasts local news music and public service announcements",
        "Flexible work schedules allow employees to choose their start and end times",
        "The archaeological museum displays pottery tools and jewelry from ancient settlements",
        "Green chemistry designs chemical products and processes that minimize toxic substances",
        "The meal prep service delivers pre-portioned ingredients with step-by-step recipe cards",
        "Air quality index readings inform residents about outdoor pollution levels each day",
        "The cohousing community shares common spaces while maintaining private living units",
        "Underwater cameras document the behavior of deep-sea creatures in their natural habitat",
        "The trade school offers certification programs in welding plumbing and electrical work",
        "The predictive text feature suggests words and phrases as the user types a message",
        "Microgrid systems provide reliable electricity to remote communities off the main grid",
        "The vintage car restoration involved sourcing original parts from multiple countries",
        "Digital accessibility guidelines ensure websites are usable by people with disabilities",
        "The cooperative extension service provides agricultural education for rural communities",
        "Touchless payment technology uses near-field communication for contactless transactions",
        "The nature documentary series captures animal behavior across six different biomes",
        "Smoke ventilation systems remove combustion gases from enclosed parking structures",
        "The pop-up art installation transformed the empty warehouse into an immersive experience",
        "Elastic computing resources scale automatically to match changing workload demands",
        "The historical walking tour visits sites significant to the founding of the city",
        "Biometric passport chips store facial features fingerprints and travel history data",
        "The shared workspace provides high-speed internet printing and meeting room access",
        "Vertical garden systems grow plants on building walls to improve urban air quality",
        "The podcast episode features an interview with a leading expert on ocean conservation",
        "Kinetic energy recovery systems capture braking energy and store it for acceleration",
        "The food pantry serves families experiencing temporary financial hardship each week",
        "Autonomous underwater vehicles map the seafloor for geological and biological research",
        "The escape room puzzle requires teamwork logic and creative problem-solving skills",
        "Solar thermal collectors heat water using concentrated sunlight for residential use",
        "The language exchange program pairs native speakers of different languages for practice",
        "Impact-resistant window glazing protects buildings during severe windstorms and hurricanes",
        "The urban farm produces fresh vegetables on a formerly vacant city lot",
        "Digital wayfinding kiosks provide interactive maps and directions inside large buildings",
        "The peer support program trains employees to provide confidential emotional support",
        "Regenerative braking systems convert kinetic energy into electrical charge for the battery",
        "The oral history project records personal accounts from surviving veterans of the conflict",
        "Low-orbit satellites provide broadband internet to underserved geographic regions",
        "The therapeutic riding program uses horseback activities to benefit riders with disabilities",
        "Biodiesel fuel can be produced from recycled cooking oil and vegetable fats",
        "The makerspace provides access to laser cutters sewing machines and woodworking tools",
        "Ambient computing integrates technology invisibly into everyday environments and objects",
        "The archival digitization project scans and indexes thousands of historical photographs",
        "Community supported agriculture members pay upfront for a season of weekly farm produce",
        "Assistive listening devices amplify sound for individuals with hearing impairments",
        "The electric ferry transports passengers across the harbor with zero direct emissions",
        "The artisan cheese maker ages wheels in a temperature-controlled cave for several months",
        "Smart building systems optimize lighting heating and security based on occupancy data",
        "The wildlife rehabilitation center treats injured birds of prey and releases them to the wild",
        "Modular furniture systems allow homeowners to reconfigure layouts without buying new pieces",
        "Mushroom cultivation uses sterilized substrate blocks in climate-controlled growing rooms",
        "The adaptive traffic signal adjusts timing based on real-time vehicle and pedestrian flow",
        "Portable water filtration systems provide clean drinking water during natural disasters",
        "The heritage grain program revives ancient wheat varieties for specialty baking purposes",
        "Indoor air quality monitors detect particulates carbon dioxide and volatile organic compounds",
        "The volunteer fire department fundraiser includes a pancake breakfast and silent auction",
        "Enzyme-based laundry detergents break down protein and starch stains at lower temperatures",
        "The public art commission selects proposals for sculptures and murals in civic spaces",
        "Cloud seeding introduces particles into clouds to encourage precipitation formation",
        "The mobile veterinary clinic travels to rural communities that lack animal care facilities",
        "Container ship scheduling coordinates berth assignments crane availability and truck pickups",
        "The community greenhouse extends the growing season for northern climate gardeners",
        "Electrostatic precipitators remove fine particles from industrial exhaust before release",
        "The storytelling festival brings together performers from oral tradition backgrounds worldwide",
        "Gravity-fed water systems deliver clean water from mountain springs to valley communities",
        "The neighborhood tool library lends power tools and garden equipment to local residents",
        "Passive house design achieves extreme energy efficiency through insulation and air sealing",
        "The floating solar farm generates electricity from panels installed on the reservoir surface",
        "Acoustic monitoring stations track whale migration patterns using underwater microphones",
        "The repair cafe helps community members fix broken electronics clothing and small appliances",
        "Rainforest canopy walkways allow researchers and visitors to observe treetop ecosystems",
        "The hydrogen fuel cell vehicle emits only water vapor from its tailpipe during operation",
        "Dark sky preserves limit artificial lighting to protect nighttime visibility of stars",
        "The social enterprise employs workers with barriers to employment and reinvests its profits",
        "Vertical oceanographic profilers measure temperature salinity and dissolved oxygen at depth",
        "The community fridge program reduces food waste by sharing surplus items with neighbors",
    ])

    # ---- Final batch to ensure broad coverage ----
    sentences.extend([
        "The blood pressure cuff provides readings that help diagnose hypertension early",
        "Crowdfunding platforms allow creators to raise money directly from supporters",
        "The coral nursery grows fragments on underwater frames for later reef transplantation",
        "Time management techniques include prioritization batching and the Pomodoro method",
        "The cheese cave maintains constant humidity and temperature for optimal aging conditions",
        "Open-source software licenses allow anyone to use modify and distribute the code freely",
        "The memorial garden provides a peaceful space for reflection and remembrance",
        "Electromagnetic induction heating rapidly warms cookware on compatible stovetop surfaces",
        "The mentoring circle brings together professionals at different career stages for dialogue",
        "Freeze-drying preserves food by removing moisture under vacuum at low temperature",
        "The community mural project invites residents to paint panels reflecting neighborhood identity",
        "Satellite imagery helps track deforestation rates across tropical regions over time",
        "The public swimming pool offers lap lanes open swim and water aerobics classes",
        "Thermographic imaging reveals hidden moisture damage behind walls and ceilings",
        "The cooperative childcare center is owned and managed by the families it serves",
        "Precision GPS guidance systems help farmers plant rows with centimeter-level accuracy",
        "The outdoor cinema screens classic films on summer evenings in the park amphitheater",
        "Desalination membrane technology has become more energy-efficient in recent years",
        "The urban beehive program places managed colonies on rooftops to support pollination",
        "Motion-sensing lighting reduces energy waste in hallways and unoccupied rooms",
        "The rare plant conservation program propagates endangered species from collected seeds",
        "Virtual museum tours allow visitors to explore galleries from anywhere in the world",
        "The meal delivery service caters to dietary restrictions including keto and vegan options",
        "Acoustic panels reduce echo and improve speech clarity in open office environments",
        "The community tool shed lends equipment for home repair and gardening projects",
        "Tidal barrage systems generate electricity from the difference between high and low tides",
        "The food safety certification program trains restaurant workers in hygiene best practices",
        "Soil biochar amendments improve water retention and microbial activity in degraded land",
        "The interactive science exhibit demonstrates principles of physics through hands-on experiments",
        "Closed-loop manufacturing recycles production waste back into the raw material supply",
        "The farm-to-table restaurant sources ingredients exclusively from producers within fifty miles",
        "Assistive robotics help elderly individuals with mobility and daily household tasks",
        "The overnight train service offers sleeping compartments for comfortable long-distance travel",
        "Phenological records track the timing of seasonal biological events like flowering and migration",
        "The community bicycle repair workshop teaches basic maintenance skills to local cyclists",
        "Phosphorescent materials absorb light energy and release a visible glow in darkness",
        "The neighborhood association organizes block parties cleanups and safety patrol programs",
        "Controlled-environment agriculture produces consistent crop yields regardless of outdoor weather",
        "The online tutoring platform matches students with subject-matter experts for live sessions",
        "Bioswale landscaping filters stormwater runoff naturally through soil and plant root systems",
    ])

    return sentences


# ---------------------------------------------------------------------------
# Projection matrix training
# ---------------------------------------------------------------------------

def _train_projection_matrix(vectorizer: HashingVectorizer) -> np.ndarray:
    """Train the projection matrix W mapping sparse features → embedding space.

    Process:
        1. Generate diverse training sentences.
        2. Compute sparse HashingVectorizer features  X_sparse  (N x D_sparse).
        3. Compute real neural embeddings               X_embed  (N x D_embed)
           using fastembed (BAAI/bge-small-en-v1.5).
        4. Solve the regularized regression problem using Ridge (L2):
               W* = argmin_W  || X_sparse @ W  -  X_embed ||^2_F + alpha * ||W||^2_F
           Ridge regression handles the underdetermined case (N << D_sparse)
           better than plain lstsq by shrinking coefficients, producing a
           more generalizable projection matrix.
        5. Cache W to disk for future sessions.

    Returns:
        W : ndarray of shape (D_sparse, D_embed)
    """
    from fastembed import TextEmbedding

    logger.info("Training SPS projection matrix (one-time operation)...")

    # Step 1: build training corpus
    corpus = _build_training_corpus()
    logger.info("  Training corpus size: %d sentences", len(corpus))

    # Step 2: sparse features via HashingVectorizer
    # .transform() works without .fit() — that's the whole point of hashing.
    X_sparse = vectorizer.transform(corpus)           # sparse CSR matrix (N, 16384)

    # Step 3: real embeddings via fastembed
    logger.info("  Computing ground-truth embeddings with %s ...", EMBED_MODEL)
    model = TextEmbedding(model_name=EMBED_MODEL)
    # fastembed returns a generator; materialise it as an array.
    X_embed = np.array(list(model.embed(corpus)), dtype=np.float32)  # (N, 384)
    logger.info("  Embedding shape: %s", X_embed.shape)

    # Step 4: solve via Ridge regression (L2 regularization).
    # The system is underdetermined (N samples << D_sparse=16384 features),
    # so plain lstsq produces an under-regularized W that overfits.
    # Ridge with alpha=1.0 shrinks coefficients and generalises better.
    logger.info("  Fitting Ridge regression for W (%d x %d) ...", X_sparse.shape[1], X_embed.shape[1])
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(X_sparse, X_embed)
    # Ridge stores coefficients as (n_targets, n_features); we need (n_features, n_targets).
    W = ridge.coef_.T.astype(np.float32)
    logger.info("  Projection matrix trained.  Shape: %s", W.shape)

    # Step 5: cache to disk
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(MATRIX_CACHE_PATH), W=W)
    logger.info("  Saved projection matrix to %s", MATRIX_CACHE_PATH)

    return W


def _load_or_train_projection_matrix(vectorizer: HashingVectorizer) -> np.ndarray:
    """Load the cached projection matrix, or train it if missing."""
    if MATRIX_CACHE_PATH.exists():
        logger.info("Loading cached projection matrix from %s", MATRIX_CACHE_PATH)
        data = np.load(str(MATRIX_CACHE_PATH))
        W = data["W"]
        # Validate shape: should be (N_SPARSE_FEATURES, embed_dim)
        if W.shape[0] != N_SPARSE_FEATURES:
            logger.warning(
                "Cached matrix shape %s doesn't match N_SPARSE_FEATURES=%d. Retraining.",
                W.shape, N_SPARSE_FEATURES,
            )
            return _train_projection_matrix(vectorizer)
        logger.info("  Loaded W with shape %s", W.shape)
        return W

    return _train_projection_matrix(vectorizer)


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

@JITSearch.register("projection")
class SemanticProjectionSearch(SearchStrategy):
    """Semantic Projection Search (SPS).

    Maps cheap sparse text features through a learned linear projection to
    approximate neural-embedding space, enabling semantic search at near-
    lexical speed with no corpus-specific pre-processing.

    The projection matrix is trained once (lazily, on first use) and cached
    to disk.  Subsequent instantiations load from cache in milliseconds.
    """

    name = "projection"

    def __init__(self, *, alpha: float = 0.3) -> None:
        # The HashingVectorizer is corpus-independent: no .fit() step needed.
        # This is crucial for JIT scenarios where the document set is unknown
        # ahead of time.
        self._vectorizer = HashingVectorizer(
            n_features=N_SPARSE_FEATURES,
            ngram_range=(1, 2),        # unigrams + bigrams for phrase semantics
            lowercase=True,
            strip_accents="unicode",
            alternate_sign=True,       # reduces hash collision impact
            norm="l2",                 # L2-normalise each row
        )

        # Lazily load/train the projection matrix.
        self._W: np.ndarray | None = None

        # BM25 component for hybrid score fusion.
        # Import here to avoid circular imports (both modules register with JITSearch).
        from jit_search.lexical import LexicalSearch
        self._lexical = LexicalSearch()

        # Fusion weight: final = alpha * sps_score + (1 - alpha) * bm25_score.
        self._alpha = alpha

    @property
    def W(self) -> np.ndarray:
        """Projection matrix, loaded lazily on first access."""
        if self._W is None:
            self._W = _load_or_train_projection_matrix(self._vectorizer)
        return self._W

    def _project(self, texts: list[str]) -> np.ndarray:
        """Compute projected embeddings for a batch of texts.

        Pipeline:
            texts → HashingVectorizer → sparse (N, 16384)
                  → @ W               → dense  (N, embed_dim)
                  → L2 normalise      → unit vectors for cosine similarity
        """
        # Sparse features (no fitting needed — this is what makes SPS "JIT").
        X_sparse = self._vectorizer.transform(texts)  # CSR (N, 16384)

        # Project into approximate embedding space.
        # For efficiency: sparse @ dense is handled natively by scipy/numpy.
        projected = X_sparse @ self.W                  # dense (N, embed_dim)

        # L2-normalise so that dot product == cosine similarity.
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        # Guard against zero-norm vectors (empty/degenerate texts).
        norms = np.maximum(norms, 1e-10)
        projected = projected / norms

        return projected.astype(np.float32)

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] via min-max scaling per query."""
        smin = scores.min()
        smax = scores.max()
        if smax - smin < 1e-12:
            # All scores identical — return uniform 0.5.
            return np.full_like(scores, 0.5)
        return (scores - smin) / (smax - smin)

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search documents using hybrid SPS + BM25 score fusion.

        Steps:
            1. Compute SPS scores (cosine similarity from projection).
            2. Compute BM25 scores via LexicalSearch.
            3. Normalize both to [0, 1] range (min-max per query).
            4. Fuse: final = alpha * sps_norm + (1 - alpha) * bm25_norm.
            5. Return the top-k documents by fused score.
        """
        if not documents:
            return []

        n_docs = len(documents)
        top_k = min(top_k, n_docs)

        # --- SPS scores (projection-based cosine similarity) ---
        all_texts = [query] + documents
        all_projected = self._project(all_texts)

        query_vec = all_projected[0]          # (embed_dim,)
        doc_vecs = all_projected[1:]          # (N_docs, embed_dim)

        sps_scores = doc_vecs @ query_vec     # (N_docs,)

        # --- BM25 scores ---
        # LexicalSearch.search returns top_k results; we need scores for ALL
        # documents so we request top_k=n_docs.
        bm25_results = self._lexical.search(query, documents, top_k=n_docs)
        bm25_scores = np.zeros(n_docs, dtype=np.float64)
        for r in bm25_results:
            bm25_scores[r.index] = r.score

        # --- Min-max normalize both to [0, 1] ---
        sps_norm = self._min_max_normalize(sps_scores)
        bm25_norm = self._min_max_normalize(bm25_scores)

        # --- Fuse scores ---
        fused = self._alpha * sps_norm + (1.0 - self._alpha) * bm25_norm

        # Get top-k indices (partial sort for efficiency on large N).
        if top_k < n_docs:
            top_indices = np.argpartition(fused, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(fused[top_indices])[::-1]]
        else:
            top_indices = np.argsort(fused)[::-1]

        return [
            SearchResult(
                index=int(idx),
                score=float(fused[idx]),
                document=documents[idx],
            )
            for idx in top_indices
        ]
