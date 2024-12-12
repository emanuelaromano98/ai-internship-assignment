# AI Internship Assignment 
**Task Description:**
Use AI tools to map this market, identify companies with permitted disposal assets by state, and create a framework for analyzing their ownership (public vs. private).

**Relevant Output Material:**
*   [Images](https://github.com/emanuelaromano98/ai-internship-assignment/tree/master/Visualizations)
*   [Tables (CSV/Excel Files)](https://github.com/emanuelaromano98/ai-internship-assignment/tree/master/Company%20List%20Tables)

**Approach to Analysis:**
1. Gathered the list of Medical/Biohazardous Waste Incinerators and Commercial Autoclave Facilities using the [EPA website](https://iwaste.epa.gov/treatment-disposal-facilities) API 
2. Used OpenAI, Perplexity and Gemini APIs to request:
    - If the company was public or private ('Public', 'Private')
    - Private Equity backers
    - Company website
    - If the company is relevant in the Medical Waste Management industry ('Relevant', 'Not relevant')
3. Used Google Custom Search API to check the Google results counts from running the below query: 
    - `"{company name} medical waste management facility"`
    - The idea is that more matches imply that the company is more relevant for the given industry
4. The response from OpenAI was kept as base reference for final values, as it was the most comprehensive
5. For each facility, the final result for the following columns was determined by selecting the most frequent response among the three AI tools:
    - Status ('Public', 'Private')
    - Relevance ('Relevant', 'Not Relevant')
6. Merged the information gathered from the EPA website with the results from querying the AI tools, to get a table containing all the information
7. Filtered the table for companies classified as Relevant and sorted (in descending order) the table using result counts obtained via Google Custom Search API
8. Used the available information to create maps for visualization


