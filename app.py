from flask import Flask, render_template, request, redirect, url_for, session
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
import os
import re
import plotly.graph_objects as go

app = Flask(__name__)

# -------------------------
# Load Likes Model Pickle
# -------------------------
pickle_path = r"C:\Users\HP\youtube_prediction_project\models\likes_model.pkl"
with open(pickle_path, "rb") as f:
    likes_model = pickle.load(f)
print("Likes model loaded successfully!")

# -------------------------
# ROUTES
# -------------------------
# HOME PAGE
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------
# VIEW FORM
# -------------------------
@app.route('/view_form1')
def view_form():
    return render_template('view_form1.html')

# Likes form
@app.route('/like_form1')
def like_form():
    return render_template('like_form1.html')

# Predict likes
@app.route('/predict_likes', methods=['POST'])
def predict_likes_route():
    try:
        # Features must match the model training
        features = ['views', 'dislikes', 'comment_count', 'category_id',
                    'No_tags', 'desc_len', 'len_title', 'publish_weekday']

        # Get form data
        new_video = {f: int(request.form[f]) for f in features}

        # Convert to 2D array for prediction
        input_array = np.array([[new_video[f] for f in features]])

        # Predict
        predicted_likes = int(round(likes_model.predict(input_array)[0]))

        # Render result page
        return render_template('like_result.html', prediction=predicted_likes)

    except Exception as e:
        return f"Error during prediction: {e}"

import re

# ===========================
# 1. YouTube Categories (IDs 1-44), each with ~20 tags
# ===========================
category_tags = {
    1: ["film", "animation", "cinema", "short film", "trailer", "movie review", "movie scene", "drama", "action", "thriller",
        "romantic", "horror", "documentary", "classic", "indie", "film festival", "behind the scenes", "director", "cinematic", "movie making"],
    2: ["cars", "autos", "vehicles", "car review", "motorcycles", "car maintenance", "test drive", "auto news", "luxury cars", "sports cars",
        "car mods", "driving tips", "classic cars", "electric cars", "hybrid cars", "road test", "car gadgets", "vehicle tech", "car DIY", "mechanics"],
    10: ["music", "song", "album", "playlist", "music video", "cover song", "remix", "live performance", "concert", "pop",
         "rock", "hip hop", "jazz", "classical", "EDM", "rap", "instrumental", "singer", "band", "soundtrack"],
    15: ["pets", "animals", "puppy", "kitten", "pet care", "pet training", "animal rescue", "exotic pets", "pet grooming", "funny pets",
         "dog training", "cat care", "pet adoption", "cute dogs", "cute cats", "pet DIY", "animal behavior", "pet toys", "pet health", "pet tricks"],
    17: ["sports", "fitness", "workout", "training", "gym", "exercise", "weight loss", "cardio", "strength", "bodybuilding","cricket","win",
         "athlete", "soccer", "basketball", "tennis", "yoga", "pilates", "home workout", "fitness tips", "training drills", "sports highlights"],
    18: ["short movies", "short film", "mini film", "short cinema", "trailer", "animated short", "short animation", "short story", "film clip",
         "indie short", "short documentary", "drama short", "action short", "comedy short", "horror short", "film review short", "movie snippet", "cinema short", "short production", "short film festival"],
    19: ["travel", "adventure", "vlog", "tour", "destination", "holiday", "vacation", "nature", "culture", "city tour",
         "backpacking", "itinerary", "local food", "exploration", "journey", "beach", "mountains", "travel guide", "travel tips", "travel vlog"],
    20: ["gaming", "gameplay", "letsplay", "walkthrough", "strategy", "multiplayer", "singleplayer", "challenge", "boss fight", "esports",
         "speedrun", "PC games", "console games", "mobile games", "top games", "game tips", "gaming highlights", "gaming news", "gaming vlog", "game tutorial"],
    21: ["vlogging", "personal vlog", "daily vlog", "storytime", "day in the life", "lifestyle vlog", "adventure vlog", "home vlog", "weekend vlog", "vlogger",
         "morning routine", "evening routine", "storytelling", "life update", "college vlog", "fun moments", "vlog tips", "daily adventures", "vlog series", "vlog tutorial"],
    22: ["people", "blogs", "personal vlog", "storytime", "daily vlog", "life update", "family vlog", "college vlog", "student vlog", "work vlog",
         "home vlog", "day vlog", "vlogger", "morning routine", "evening routine", "video blog", "vlog ideas", "vlogger diary", "personal journey", "lifestyle vlog"],
    23: ["comedy", "funny", "humor", "skit", "parody", "standup", "satire", "pranks", "jokes", "viral",
         "funny fails", "reaction", "funny moments", "comedy sketch", "funny clips", "humor video", "funny animals", "comic", "fun video", "laugh"],
    24: ["entertainment", "movies", "TV shows", "music", "celebrities", "pop culture", "gaming", "funny videos", "vlog", "viral",
         "trivia", "reviews", "events", "talk show", "behind the scenes", "interviews", "live show", "drama", "comedy", "skits"],
    25: ["news", "politics", "current events", "breaking news", "analysis", "opinion", "discussion", "world news", "local news", "government",
         "elections", "policy", "report", "journalism", "media", "news update", "debate", "hot topic", "political news", "news commentary"],
    26: ["howto", "DIY", "style", "tutorial", "fashion", "makeup", "beauty tips", "home tips", "crafts", "guide",
         "step by step", "skills", "creative ideas", "learning", "life hacks", "project", "styling", "instruction", "lesson", "tips"],
    27: ["education", "learning", "study", "tutorial", "lesson", "exam tips", "online course", "academic help", "lecture", "class notes",
         "study hacks", "knowledge", "career advice", "student tips", "school tips", "study guide", "learning tips", "educational content", "learning strategies", "education vlog"],
    28: ["science", "technology", "research", "experiments", "STEM", "innovation", "gadgets", "tech review", "tech tutorial", "coding",
         "AI", "machine learning", "robotics", "space", "physics", "chemistry", "biology", "future tech", "scientific discovery", "science experiment"],
    29: ["nonprofit", "activism", "charity", "social cause", "fundraising", "volunteer", "community service", "awareness", "campaign", "ngo",
         "donation", "social impact", "humanitarian", "support", "charitable", "project", "advocacy", "movement", "social work", "benefit"],
    30: ["movies", "film", "cinema", "blockbuster", "trailer", "drama", "action", "thriller", "romantic", "horror",
         "documentary", "classic", "indie", "movie review", "film festival", "behind the scenes", "director", "short film", "animated film", "cinematic"],
    31: ["anime", "animation", "manga", "anime review", "anime clip", "anime scene", "cartoon", "animated series", "anime short", "anime film",
         "animation tutorial", "anime analysis", "anime discussion", "anime news", "anime music", "cosplay", "anime fan", "anime highlights", "anime character", "anime episode"],
    32: ["action", "adventure", "stunts", "fight scene", "hero", "mission", "thriller", "exploration", "journey", "battle",
         "combat", "epic", "action movie", "action trailer", "adventure film", "action scene", "adventure short", "thrilling", "action vlog", "stunt"],
    33: ["classics", "classic movie", "old film", "cinema classic", "vintage movie", "classic drama", "classic action", "classic comedy", "classic review", "timeless film",
         "cinema history", "movie heritage", "classic scene", "golden age", "classic trailer", "classic analysis", "film retro", "classic collection", "classic cinema", "classic performance"],
    34: ["comedy", "funny", "humor", "parody", "skit", "standup", "viral", "pranks", "jokes", "satire",
         "funny fails", "reaction", "comedy sketch", "humor video", "funny clips", "fun video", "laugh", "comedy vlog", "comic", "funny moments"],
    35: ["documentary", "real story", "informative", "educational", "history", "culture", "nature", "environment", "biography", "society",
         "facts", "events", "insight", "investigation", "reportage", "interview", "storytelling", "exploration", "documentary short", "documentary series"],
    36: ["drama", "emotional", "story", "character", "plot", "series", "episode", "movie scene", "theatre", "romance",
         "thriller", "action", "suspense", "family", "conflict", "storytelling", "film", "drama clip", "dramatic", "performance"],
    37: ["family", "kids", "children", "parenting", "fun", "education", "games", "activities", "family vlog", "home",
         "family moments", "family tips", "family entertainment", "child care", "learning", "toys", "family guide", "parent tips", "family time", "family fun"],
    38: ["foreign", "international", "foreign film", "international cinema", "world cinema", "foreign review", "foreign trailer", "foreign short", "culture", "subtitled",
         "foreign drama", "foreign action", "foreign comedy", "foreign animation", "foreign movie", "world film", "foreign clip", "global cinema", "foreign story", "foreign entertainment"],
    39: ["horror", "scary", "ghost", "thriller", "horror movie", "haunted", "paranormal", "creepy", "frightening", "suspense",
         "horror short", "horror scene", "horror clip", "terror", "fear", "monster", "spooky", "horror review", "horror trailer", "scary moments"],
    40: ["sci-fi", "fantasy", "space", "alien", "future", "technology", "magic", "fiction", "adventure", "epic",
         "sci-fi movie", "fantasy movie", "sci-fi trailer", "fantasy trailer", "sci-fi clip", "fantasy clip", "sci-fi short", "fantasy short", "sci-fi series", "fantasy series"],
    41: ["thriller", "suspense", "crime", "mystery", "action", "drama", "psychological", "chase", "detective", "investigation",
         "crime thriller", "murder mystery", "intense", "plot twist", "thriller scene", "thriller clip", "thriller short", "suspenseful", "thriller movie", "suspense series"],
    42: ["shorts", "short video", "mini clip", "quick tutorial", "fun short", "short comedy", "short drama", "quick guide", "mini vlog", "short music",
         "short film", "animated short", "short animation", "short review", "short snippet", "short highlights", "short story", "mini movie", "short clip", "viral short"],
    43: ["shows", "TV show", "web series", "episode", "season", "entertainment", "drama", "comedy", "reality show", "talk show",
         "live show", "game show", "quiz show", "talent show", "series", "performance", "episode guide", "episode review", "show highlights", "show clip"],
    44: ["trailers", "movie trailer", "film trailer", "short trailer", "teaser", "trailer review", "blockbuster trailer", "action trailer", "horror trailer", "sci-fi trailer",
         "romantic trailer", "comedy trailer", "animation trailer", "short film trailer", "trailer breakdown", "official trailer", "trailer analysis", "upcoming trailer", "new trailer", "movie teaser"],
}

# ===========================
# 2. Keyword Tagging (20 keywords, each with 20 tags)
# ===========================
keyword_tags_mapping = {
    "tutorial": ["how to", "guide", "step by step", "DIY tutorial", "learn", "education", "training", "video tutorial", "beginner guide", "lesson",
                 "class", "course", "walkthrough", "tutorial tips", "instruction guide", "tutorial tricks", "tutorial hacks", "learning tips", "learning guide", "full tutorial"],
    "python": ["python", "python tutorial", "coding", "programming", "learn python", "python code", "python project", "python for beginners", "python script", "python programming",
               "python examples", "python exercises", "python course", "python lessons", "python training", "python challenge", "python project ideas", "python concepts", "python automation", "python coding tips"],
    "ai": ["AI", "artificial intelligence", "machine learning", "deep learning", "neural networks", "AI tutorial", "ML tutorial", "AI project", "ML project", "AI coding",
           "AI concepts", "AI programming", "ML concepts", "ML coding", "AI demo", "ML demo", "AI guide", "ML guide", "AI learning", "ML learning"],
    "gaming": ["gaming", "gameplay", "letsplay", "walkthrough", "strategy", "multiplayer", "singleplayer", "challenge", "boss fight", "esports",
               "speedrun", "PC games", "console games", "mobile games", "top games", "game tips", "gaming highlights", "gaming news", "gaming vlog", "game tutorial"],
    "music": ["music", "song", "album", "playlist", "music video", "cover song", "remix", "live performance", "concert", "pop",
              "rock", "hip hop", "jazz", "classical", "EDM", "rap", "instrumental", "singer", "band", "soundtrack"],
    "funny": ["funny", "comedy", "humor", "skit", "parody", "standup", "satire", "pranks", "jokes", "viral",
              "funny fails", "reaction", "funny moments", "comedy sketch", "funny clips", "humor video", "funny animals", "comic", "fun video", "laugh"],
    "review": ["review", "product review", "comparison", "opinions", "first impressions", "unboxing", "demo", "ratings", "testing", "analysis",
               "evaluation", "feedback", "recommendation", "hands-on", "review guide", "review tips", "review session", "review series", "product demo", "expert review"],
    "fitness": ["fitness", "workout", "training", "gym", "exercise", "weight loss", "cardio", "strength", "bodybuilding", "home workout",
                "fitness tips", "athlete", "sports", "personal trainer", "fitness challenge", "yoga", "pilates", "nutrition", "fitness motivation", "fitness plan"],
    "travel": ["travel", "vlog", "tour", "holiday", "vacation", "adventure", "destination", "travel guide", "travel tips", "city tour",
               "nature", "beach", "mountains", "backpacking", "culture", "local food", "itinerary", "travel vlog", "exploration", "journey"],
    "food": ["food", "recipe", "cooking", "baking", "kitchen", "meal", "dish", "culinary", "food vlog", "easy recipe",
             "healthy", "snacks", "dessert", "lunch", "dinner", "breakfast", "chef", "tutorial", "cooking tips", "food guide"],
    "education": ["education", "learning", "study", "tutorial", "lesson", "exam tips", "online course", "academic help", "lecture", "class notes",
                  "study hacks", "learning tips", "knowledge", "career advice", "student tips", "school tips", "study guide", "educational content", "learning strategies", "education vlog"],
    "science": ["science", "technology", "research", "experiments", "STEM", "innovation", "gadgets", "tech review", "tech tutorial", "coding",
                "AI", "machine learning", "robotics", "space", "physics", "chemistry", "biology", "future tech", "scientific discovery", "science experiment"],
    "vlog": ["vlog", "daily vlog", "personal vlog", "storytime", "life update", "adventure vlog", "daily life", "weekend vlog", "vlogger", "lifestyle vlog",
             "storytelling", "vlog ideas", "vlog tutorial", "vlogging guide", "vlogger diary", "video diary", "daily stories", "fun vlog moments", "vlog trends", "vlog entertainment"],
    "pets": ["pets", "animals", "puppy", "kitten", "pet care", "pet training", "animal rescue", "exotic pets", "pet grooming", "funny pets",
             "dog training", "cat care", "pet adoption", "cute dogs", "cute cats", "pet DIY", "animal behavior", "pet toys", "pet health", "pet tricks"],
    "technology": ["technology", "gadgets", "innovation", "AI", "coding", "programming", "tech review", "future tech", "tech tutorial", "robotics",
                   "machine learning", "STEM", "tech news", "tech guide", "tech demo", "tech project", "tech hacks", "coding tips", "technology learning", "tech content"],
    "fashion": ["fashion", "style", "clothing", "outfit", "lookbook", "fashion tips", "haul", "makeup", "beauty", "trend",
                "shopping", "wardrobe", "styling", "fashion vlog", "street style", "fashion review", "OOTD", "accessories", "fashion ideas", "fashion tutorial"],
    "art": ["art", "painting", "drawing", "sketch", "digital art", "artist", "art tutorial", "artwork", "creative", "illustration",
            "design", "art tips", "art process", "art techniques", "art project", "art vlog", "art demo", "painting tutorial", "drawing tutorial", "art ideas"],
    "movies": ["movies", "film", "cinema", "short film", "trailer", "animation", "movie review", "movie scene", "drama", "action",
               "thriller", "romantic", "horror", "documentary", "classic", "indie", "cinema news", "film festival", "movie making", "behind the scenes"],
}

# ===========================
# 3. Stopwords
# ===========================
stop_words = set(["for", "in", "the", "a", "an", "and", "of", "to", "on", "with", "by", "at", "from", "is", "are", "it", "new"])

# ===========================
# 4. Simple Tokenizer
# ===========================
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# ===========================
# 5. Tag Recommender
# ===========================
category_name_to_id = {
    "Film & Animation": 1,
    "Autos & Vehicles": 2,
    "Music": 10,
    "Pets & Animals": 15,
    "Sports": 17,
    "Short Movies": 18,
    "Travel & Events": 19,
    "Gaming": 20,
    "Videoblogging": 21,
    "People & Blogs": 22,
    "Comedy": 23,
    "Entertainment": 24,
    "News & Politics": 25,
    "Howto & Style": 26,
    "Education": 27,
    "Science & Technology": 28,
    "Nonprofits & Activism": 29,
    "Movies": 30,
    "Anime/Animation": 31,
    "Action/Adventure": 32,
    "Classics": 33,
    "Comedy (alternate)": 34,
    "Documentary": 35,
    "Drama": 36,
    "Family": 37,
    "Foreign": 38,
    "Horror": 39,
    "Sci-Fi/Fantasy": 40,
    "Thriller": 41,
    "Shorts": 42,
    "Shows": 43,
    "Trailers": 44
}

def recommend_tags(category_name, video_title, top_n=15):
    tags = []

    # Convert category name → ID
    category_id = category_name_to_id.get(category_name)

    # 1️⃣ Add category-specific tags
    if category_id:
        tags.extend(category_tags.get(category_id, []))

    # 2️⃣ Tokenize title
    words = simple_tokenize(video_title)
    keywords = [w for w in words if w not in stop_words]

    # 3️⃣ Keyword → tag expansion
    for kw in keywords:
        for key, tag_list in keyword_tags_mapping.items():
            if key in kw:
                tags.extend(tag_list)
        tags.append(kw)

    # 4️⃣ Deduplicate
    seen = set()
    final_tags = []
    for t in tags:
        if t not in seen:
            final_tags.append(t)
            seen.add(t)

    # 5️⃣ Remove spaces
    final_tags = [t.replace(" ", "") for t in final_tags]

    return final_tags[:top_n]



# TAGS FORM
# -------------------------
@app.route('/recommend_form1')
def recommend_form():
    return render_template('recommend_form1.html')

@app.route('/recommend_tags', methods=['POST'])
def recommend_tags_route():
    try:
        category_name = request.form.get('video_category')
        video_title = request.form.get('video_title')

        if not category_name or not video_title:
            return "Category and title are required", 400

        tags = recommend_tags(
            category_name=category_name,
            video_title=video_title,
            top_n=15
        )

        return render_template(
            'recommend_result.html',
            tags=tags,
            title=video_title,
            category=category_name
        )

    except Exception as e:
        return f"Error in tag recommendation: {e}", 400




app.secret_key = "youtube-secret"

BASE = r"C:\Users\HP\youtube_prediction_project\models"

# ---- Load models ----
models = {
    "rf": pickle.load(open(f"{BASE}/views_rf.pkl", "rb")),
    "gb": pickle.load(open(f"{BASE}/views_gbr.pkl", "rb")),
    "xgb": pickle.load(open(f"{BASE}/views_xgb.pkl", "rb")),
    "lgb": pickle.load(open(f"{BASE}/views_lgb.pkl", "rb")),
}

imputer = pickle.load(open(f"{BASE}/imputer.pkl", "rb"))
pt = pickle.load(open(f"{BASE}/power_transformer.pkl", "rb"))
numerical_cols = pickle.load(open(f"{BASE}/numerical_cols.pkl", "rb"))
feature_cols = pickle.load(open(f"{BASE}/feature_columns.pkl", "rb"))

def predict_all_models(input_dict):
    df = pd.DataFrame([input_dict])

    # add missing columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    # preprocessing
    df[numerical_cols] = imputer.transform(df[numerical_cols])
    df[numerical_cols] = np.log1p(df[numerical_cols])
    df[numerical_cols] = pt.transform(df[numerical_cols])

    predictions = {}
    for key, model in models.items():
        log_pred = model.predict(df)[0]
        predictions[key] = int(np.expm1(log_pred))

    return predictions

#VIEWS FORM
@app.route("/predict_views", methods=["POST"])
def predict_views_route():

    input_data = {
        "duration": int(request.form["duration"]),
        "likes": int(request.form["likes"]),
        "dislikes": int(request.form["dislikes"]),
        "comment_count": int(request.form["comment_count"]),
        "category_id": int(request.form["category_id"]),
        "publish_day": int(request.form["publish_day"]),
        "publish_month": int(request.form["publish_month"]),
        "publish_hour": int(request.form["publish_hour"]),
        "trending_day": int(request.form.get("trending_day", 0)),
        "trending_month": int(request.form.get("trending_month", 0)),
        "subscribers": int(request.form["subscribers"]),
        "total_videos": int(request.form["total_videos"]),
    }

    selected_model = request.form["model"]   # rf / gb / xgb / lgb

    # 🔹 predict using ALL models
    all_predictions = predict_all_models(input_data)

    # 🔹 store for insights (USE CONSISTENT KEYS)
    session["last_input"] = input_data
    session["last_predictions"] = all_predictions
    session["selected_model"] = selected_model

    # 🔹 REDIRECT instead of rendering (PRG pattern)
    return redirect(url_for("view_result_page"))

@app.route("/view_result_page")
def view_result_page():
    last_input = session.get("last_input")
    last_predictions = session.get("last_predictions")
    selected_model = session.get("selected_model")

    if not last_input or not last_predictions:
        return redirect(url_for("view_form"))

    model_names = {
        "rf": "Random Forest",
        "gb": "Gradient Boosting",
        "xgb": "XGBoost",
        "lgb": "LightGBM"
    }

    final_prediction = int(last_predictions[selected_model])

    return render_template(
        "view_result.html",
        result=final_prediction,
        model_name=model_names[selected_model]
    )



#INSIGHTS
@app.route("/show_insights")
def show_insights():

    new_video = session.get("last_input")
    result = session.get("last_predictions")

    if not new_video or not result:
         return render_template(
            "show_insights_form1.html",
            no_data=True
        )

    # ---------------------------
    # INSIGHT 1: HEATMAP
    # ---------------------------

    predicted_views = np.mean(list(result.values()))

    days = np.arange(1,8)
    hours = np.arange(
        new_video["publish_hour"] - 3,
        new_video["publish_hour"] + 4
    )

    data = []
    for d in days:
        for h in hours:
            score = predicted_views * \
                    (1 + ((d - new_video["publish_day"]) * 0.05)) * \
                    (1 + ((h - new_video["publish_hour"]) * 0.03))
            data.append([d, h, score])

    df = pd.DataFrame(data, columns=["publish_day", "publish_hour", "views"])
    heatmap_df = df.pivot(index="publish_day", columns="publish_hour", values="views")

    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Publish Hour", y="Publish Day", color="Predicted Views")
    )
    fig.update_traces(
        hovertemplate=
        "Publish Day: %{y}<br>"
        "Publish Hour: %{x}<br>"
        "Predicted Views: %{z:.0f}<extra></extra>"
    )

    fig.update_layout(
        height=450,
        title_font_size=24,
        margin=dict(l=80, r=30, t=80, b=80),
        coloraxis_colorbar=dict(
            tickformat=","
        ),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True)
    )
    heatmap_html = fig.to_html(full_html=False)

    # -------------------------------
    # INSIGHT 2: Average Views by Hour
    # -------------------------------
    hour_views = df.groupby("publish_hour")["views"].mean().round(0).astype(int).reset_index()

    fig_hour = px.line(
        hour_views,
        x="publish_hour",
        y="views",
        markers=True,
        labels={
            "publish_hour": "Publish Hour (0–23) Hour",
            "views": "Average Predicted Views"
        }
    )

    # Add color and line width
    fig_hour.update_traces(
        line=dict(color="gold", width=3),  # red line, thicker
        marker=dict(size=8, color="gold")  # markers in same color
    )


    fig_hour.update_layout(
        width=700,
        height=400,
        title_font_size=22,
        yaxis_tickformat=",",
        margin=dict(l=80, r=30, t=70, b=70),
        xaxis=dict(
            title="Publish Hour (0–23)",
            tickmode="linear",
            automargin=True
        ),
        yaxis=dict(
            title="Average Predicted Views",
            automargin=True
        )
    )

    hour_chart_html = fig_hour.to_html(full_html=False)


    # -------------------------------
    # INSIGHT 3: Average Views by Day
    # -------------------------------
    # Remap days: Sun=0, Sat=1, Mon=2 ... Fri=6
    day_map = {7: 0, 6: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    df["publish_day"] = df["publish_day"].map(day_map)


    day_views = df.groupby("publish_day")["views"].mean().round(0).astype(int).reset_index()

    fig_day = px.line(
        day_views,
        x="publish_day",
        y="views",
        markers=True,
        labels={
            "publish_day": "Publish Day (Sun=0-Sat=6) Day",
            "views": "Average Predicted Views"
        }
    )

    # Add color and marker style
    fig_day.update_traces(
        line=dict(color="darkviolet", width=3),
        marker=dict(size=8, color="darkviolet")
    )

    fig_day.update_layout(
        width=700,
        height=400,
        title_font_size=22,
        yaxis_tickformat=",",
        margin=dict(l=80, r=30, t=70, b=70),
        xaxis=dict(
            title="Publish Day (Sun = 0 → Sat = 6)",
            automargin=True
        ),
        yaxis=dict(
            title="Average Predicted Views",
            automargin=True
        )
    )

    day_chart_html = fig_day.to_html(full_html=False)

    # -------------------------------
    # INSIGHT 4: Model Comparison (Views)
    # -------------------------------

    # Model display names
    model_names = {
        "rf": "Random Forest",
        "gb": "Gradient Boosting",
        "xgb": "XGBoost",
        "lgb": "LightGBM"
    }

    # Example accuracy values (replace with real ones if available)
    accuracies = {
        "rf": 0.9826,
        "gb": 0.9764,
        "xgb": 0.9028,
        "lgb": 0.9002
    }

    # Required order
    custom_order = [
        "Gradient Boosting",
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ]

    # Prepare DataFrame from session predictions (past data)
    df_models = pd.DataFrame({
        "Model": [model_names[m] for m in result.keys()],
        "Views": [result[m] for m in result.keys()],
        "Accuracy": [accuracies[m] for m in result.keys()]
    })

    #APPLY ORDER (THIS WAS MISSING)
    df_models["Model"] = pd.Categorical(
        df_models["Model"],
        categories=custom_order,
        ordered=True
    )
    df_models = df_models.sort_values("Model")

    # Label to show views count on bars
    df_models["Views_Label"] = df_models["Views"].apply(lambda x: f"{x:,}")

    # Create vertical bar chart
    fig_model = px.bar(
        df_models,
        x="Model",
        y="Views",
        text="Views_Label",
        color_discrete_sequence=["orange"],
        hover_data={
            "Model": True,
            "Accuracy": ":.2%",
            "Views": False          # hide views in hover (already on bars)
        }
    )

    # Text inside bars
    fig_model.update_traces(
        textposition="inside",
        insidetextfont=dict(color="black")
    )

    # Layout styling (match other insights)
    fig_model.update_layout(
        width=700,
        height=350,
        xaxis_title="ML Model",
        yaxis_title="Predicted Views",
        yaxis_tickformat=",",
        title_font_size=22,
        margin=dict(t=60, b=50, l=30, r=30),
        hoverlabel=dict(
            bgcolor="white",
            font_color="black"
        )
    )

    model_chart_html = fig_model.to_html(full_html=False)


    # -------------------------------
    # INSIGHT 5: Engagement Breakdown
    # -------------------------------

    likes = new_video["likes"]
    comments = new_video["comment_count"]
    dislikes = new_video["dislikes"]

    df_engage = pd.DataFrame({
        "Engagement": ["Likes", "Comments", "Dislikes"],
        "Count": [likes, comments, dislikes]
    })

    fig_engage = px.pie(
        df_engage,
        names="Engagement",
        values="Count",
        hole=0.4,
        color_discrete_sequence=["#2ECC71", "#3498DB", "#E67E22"]
    )

    fig_engage.update_layout(
        width=700,
        height=350,
        title_font_size=22,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    fig_engage.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>Count: %{value:,}<extra></extra>",
        domain=dict(x=[0.02, 0.98], y=[0.02, 0.98])
    )

    #Convert to HTML (VERY IMPORTANT)
    engagement_chart_html = fig_engage.to_html(full_html=False)

    return render_template(
        "show_insights_form1.html",
        heatmap_html=heatmap_html,
        hour_chart_html=hour_chart_html,
        day_chart_html=day_chart_html,
        model_chart_html=model_chart_html,
        engagement_chart_html=engagement_chart_html,
        no_data=False
    )



    


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
