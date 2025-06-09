import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np



# --- Miary podobieństwa ---
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_distance(x, y):
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 1.0
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def chi_square_distance(x, y):
    return np.sum((x - y) ** 2 / (x + y + 1e-10))

def normalized_euclidean_distance(x, y, mean, std):
    x_norm = (x - mean) / (std + 1e-10)
    y_norm = (y - mean) / (std + 1e-10)
    return np.sqrt(np.sum((x_norm - y_norm) ** 2))

# --- Główna funkcja rekomendacji ---
def get_recommendations(df, title, method='cosine', n=5,
                        use_year=True, use_duration=True, use_director=True):
    df = df.copy()
    if title not in df['title'].values:
        return [f"Film '{title}' nie został znaleziony w zbiorze."]

    features = []
    if use_year:
        df['year_scaled'] = (df['year'] - df['year'].min()) / 10
        features.append('year_scaled')
    if use_duration:
        df['duration_scaled'] = (df['duration'] - df['duration'].min()) / 100
        features.append('duration_scaled')

    if use_director:
        def director_score(row_directors, target_directors):
            if not isinstance(row_directors, list) or not isinstance(target_directors, list):
                return 15
            return 5 * len(set(target_directors) - set(row_directors))

        df['directors_list'] = df['directors'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
        target_directors = df[df['title'] == title]['directors_list'].values[0]
        df['director_score'] = df['directors_list'].apply(lambda d: director_score(d, target_directors))
        features.append('director_score')

    feature_matrix = df[features].to_numpy()
    titles = df['title'].to_numpy()
    idx = df[df['title'] == title].index[0]
    target_vector = feature_matrix[idx]

    distances = []
    if method == 'normalized_euclid':
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        for vec in feature_matrix:
            dist = normalized_euclidean_distance(target_vector, vec, mean, std)
            distances.append(dist)
    else:
        for vec in feature_matrix:
            if method == 'euclid':
                dist = euclidean_distance(target_vector, vec)
            elif method == 'chi_square':
                dist = chi_square_distance(target_vector, vec)
            else:
                dist = cosine_distance(target_vector, vec)
            distances.append(dist)

    df['distance'] = distances
    recommendations = df[df['title'] != title].sort_values('distance').head(n)
    return recommendations['title'].tolist()

# --- Wczytanie rzeczywistych danych ---
@st.cache_data
def load_data():
    df = pd.read_csv("filmtv_movies.csv")
    df['directors'] = df['directors'].fillna("").astype(str)
    return df

data = load_data()



# --- Streamlit UI ---
st.title("🎬 System rekomendacji filmów")

col11, col21 = st.columns(2)


with col11:
    st.header("🧾 Podaj parametry rekomendacji")
    film = st.text_input("Wpisz tytuł filmu:", value="Braveheart")

    method = st.selectbox("Wybierz miarę podobieństwa:",
                        options=['Podobieństwo cosinusowe', 'Podobieństwo euklidesa', 'Podobieństwa chi-kwadrat', 'Znormalizowane podobieństwo Euklidesa'])

    col1, col2, col3 = st.columns(3)
    with col1:
        use_year = st.checkbox("Weź pod uwagę rok produkcji", value=True)
    with col2:
        use_duration = st.checkbox("Weź pod uwagę długość filmu", value=True)
    with col3:
        use_director = st.checkbox("Weź pod uwagę reżysera", value=True)

    n = st.slider("Liczba rekomendacji:", min_value=1, max_value=15, value=5)

    if st.button("🔍 Pokaż rekomendacje"):
        if method == 'Podobieństwo cosinusowe':
            method = 'cosine'
        elif method == 'Podobieństwo euklidesa':
            method = 'euclid'
        elif method == 'Podobieństwa chi-kwadrat':
            method = 'chi_square'
        elif method == 'Znormalizowane podobieństwo Euklidesa':
            method = 'normalized_euclid'

        results = get_recommendations(data, film, method, n, use_year, use_duration, use_director)
        st.subheader("Polecane filmy:")
        for r in results:
            st.markdown(f"- {r}")

with col21:
    st.header("🎞️ Przeglądaj bazę filmów")

    with st.expander("🔎 Filtruj filmy"):
        selected_genre = st.multiselect("Wybierz gatunek:", options=sorted(data['genre'].dropna().unique()))
        selected_country = st.multiselect("Wybierz kraj:", options=sorted(data['country'].dropna().unique()))
        min_year, max_year = int(data['year'].min()), int(data['year'].max())
        selected_year = st.slider("Zakres lat produkcji:", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    # Filtrowanie danych
    filtered = data.copy()
    if selected_genre:
        filtered = filtered[filtered['genre'].isin(selected_genre)]
    if selected_country:
        filtered = filtered[filtered['country'].isin(selected_country)]
    filtered = filtered[(filtered['year'] >= selected_year[0]) & (filtered['year'] <= selected_year[1])]

    st.dataframe(filtered)

