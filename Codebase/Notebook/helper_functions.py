
import requests
import pandas as pd
from time import sleep
from openai import OpenAI
from dotenv import load_dotenv
import os
import folium
import json

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY_1 = os.getenv('GOOGLE_API_KEY_1')
GOOGLE_API_KEY_2 = os.getenv('GOOGLE_API_KEY_2')
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# EPA API
def fetch_facilities(facility_type_ids="4,14"):
    page_num = 1
    all_facilities = []
    total_count = None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    while True:
        try:
            api_url = f"https://iwaste.epa.gov/api/facilities?facilityTypeId={facility_type_ids}&page={page_num}"
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            facilities = data.get('data', [])
            if not facilities:
                break
            if total_count is None and facilities:
                total_count = data.get('total', 0)
                print(f"Total facilities to fetch: {total_count}")
            all_facilities.extend(facilities)
            if len(all_facilities) >= total_count:
                break
            page_num += 1
            sleep(0.5)
        except requests.exceptions.RequestException as e:
            break
        except ValueError as e:
            print(f"Error parsing JSON on page {page_num}: {e}")
            break
    if all_facilities:
        df = pd.DataFrame(all_facilities)
        df['facilitySubtypeIds'] = df['facilitySubtypeIds'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
        df = df.drop(columns=['location', 'facilityTypes', 'facilityTypeIds', 'hasDfr', 'totalCount'])
        df = df.replace(['N/A', 'Not Available', 'Not available']  , None)
        print(f"Fetched {len(df)} facilities")
        return df
    else:
        print("No facilities found")
        return None
    
# OpenAI API
def call_openai_api(query, client=OpenAI(api_key=OPENAI_API_KEY), model='gpt-4'):
  completion = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "user",
       "content": query}
    ]
  )
  return completion.choices[0].message.content


# Perplexity API
def call_perplexity_api(query, url='https://api.perplexity.ai/chat/completions', model="llama-3.1-sonar-small-128k-online"):
  payload = {
      "messages": [
          {
              "role": "user",
              "content": query
          }
      ],
      "model": model,
      "temperature": 0.2,
      "top_p": 0.9,
      "search_domain_filter": ["perplexity.ai"],
      "return_images": False,
      "return_related_questions": False,
      "search_recency_filter": "month",
      "top_k": 0,
      "stream": False,
      "presence_penalty": 0,
      "frequency_penalty": 1
  }
  headers = {
      "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
      "Content-Type": "application/json"
  }
  response = requests.request("POST", url, json=payload, headers=headers)
  return response.json()


def call_gemini_api(query, api_key=GEMINI_API_KEY, model="gemini-1.5-flash-latest"):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    payload = {
      "contents": [
            {
                "parts": [
                    {"text": query}
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(f"{api_url}?key={api_key}", headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")


# Google Search API
def get_google_search_results_count_first(query, api_key=GOOGLE_API_KEY_1, cx=GOOGLE_SEARCH_ENGINE_ID):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "searchInformation" in data:
            return data["searchInformation"]["totalResults"]
        
        else:
            return "No results found in response."
    else:
        return f"Error: {response.status_code}, {response.text}"
    
    
def get_google_search_results_count_second(query, api_key=GOOGLE_API_KEY_2, cx=GOOGLE_SEARCH_ENGINE_ID):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "searchInformation" in data:
            return data["searchInformation"]["totalResults"]
        
        else:
            return "No results found in response."
    else:
        return f"Error: {response.status_code}, {response.text}"
    

# Normalize company names
def normalize_company_names(df):
    df['name'] = df['name'].apply(lambda x: x.split(" / ")[0] if " / " in x else x)
    df['name'] = df['name'].apply(lambda x: x.split("/")[0] if "/" in x else x)
    df['name'] = df['name'].apply(lambda x: x.split(" - ")[0] if " - " in x else x)
    df['name'] = df['name'].apply(lambda x: "Stericycle" if 'stericycle' in x.lower() else x)
    return df


# Generate map
def generate_google_style_static_map(df, ref_col='facilitySubtypeIds', label_1='Incinerator', label_2='Autoclave', addition_to_title='', top_companies=None):
    import geopandas as gpd
    import contextily as ctx
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from matplotlib.lines import Line2D
    import pandas as pd

    if top_companies is None:
        top_companies = []

    facility_counts = df[ref_col].value_counts()
    total_count = facility_counts.get(label_1, 0) + facility_counts.get(label_2, 0)

    # Create summary table
    summary_df = pd.DataFrame({
        'Facility Type': [label_1, label_2],
        'Count': [
            facility_counts.get(label_1, 0),
            facility_counts.get(label_2, 0),
        ]
    })
    display(summary_df)

    # Generate GeoDataFrame
    df_tmp = df.copy()
    df_tmp['geometry'] = df_tmp.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df_tmp, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=3857)

    # Initialize the plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Titles for each map
    titles = [
        f"{label_1} vs {label_2} {addition_to_title}",
        f"{label_1} vs {label_2} {addition_to_title} (Highlight Top 10 Companies' Facilities)"
    ]

    # Iterate for two maps
    for idx, ax in enumerate(axes):
        color_mapping = {label_1: "red", label_2: "blue"}
        for facility_type, color in color_mapping.items():
            subset = gdf[gdf[ref_col] == facility_type]

            if idx == 1:  # Highlight top_companies
                subset['color'] = subset['name'].apply(
                    lambda x: "#32CD32" if x in top_companies else color
                )
            else:  # Default colors
                subset['color'] = color

            subset.plot(
                ax=ax,
                color=subset['color'],
                label=f"{facility_type}",
                alpha=0.6,
                marker='o',
                markersize=50
            )

        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Add title
        ax.set_title(titles[idx])


        # Add legend
        if idx == 1:  # Special legend for highlighted map
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f"Top 10 Companies' Facilities",
                       markerfacecolor='#32CD32', markersize=10),
                Line2D([0], [0], marker='o', color='w', label=f"{label_1} ({facility_counts.get(label_1, 0)})",
                       markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', label=f"{label_2} ({facility_counts.get(label_2, 0)})",
                       markerfacecolor='blue', markersize=10)
            ]
        else:  # Default legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=f"{label_1} ({facility_counts.get(label_1, 0)})",
                       markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', label=f"{label_2} ({facility_counts.get(label_2, 0)})",
                       markerfacecolor='blue', markersize=10)
            ]
        ax.legend(handles=legend_elements, loc='lower left')

    # Adjust layout and save
    plt.tight_layout()
    output_file = f"{label_1}_vs_{label_2}_{addition_to_title}_side_by_side.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()

    return output_file



def explode_columns_perplexity(df, columns):
  df_tmp = df.copy()
  citations = []
  for column in columns:
    df_tmp[column] = df_tmp[column].apply(
      lambda x: (
          x.get("choices", [{}])[0]
          .get("message", {})
          .get("content", "Unknown")
      ) if isinstance(x, dict) else "Unknown"
    )
  return df_tmp


def explode_columns_gemini(df, columns):
  df_tmp = df.copy()
  for column in columns:
    df_tmp[column] = df_tmp[column].apply(lambda x: x['candidates'][0]['content']['parts'][0]['text'] if x else None)
  return df_tmp


def normalize_columns(df, columns, values):
  df_tmp = df.copy()
  for column, value in zip(columns, values):
    df_tmp[column] = df_tmp[column].apply(lambda x: value if isinstance(x, str) and value in x else x)
  df_tmp = df_tmp.apply(lambda x: x.replace("\n", ""))
  return df_tmp



def get_mode(ref_df, dfs, col):
    df_tmp = ref_df.copy()
    status_mode = pd.concat([df[col] for df in dfs], axis=1).mode(axis=1)[0]
    df_tmp[col] = status_mode
    return df_tmp


def json_to_df(file_name):
  try:
    with open(f'{file_name}', 'r') as file:
        data = json.load(file)
    df_ai = pd.DataFrame(data)
    print(f"Successfully loaded {file_name} into a DataFrame.")
  except FileNotFoundError:
    print(f"Error: {file_name} not found. Please ensure the file exists in the current working directory.")
    df_ai = pd.DataFrame()
  return df_ai
