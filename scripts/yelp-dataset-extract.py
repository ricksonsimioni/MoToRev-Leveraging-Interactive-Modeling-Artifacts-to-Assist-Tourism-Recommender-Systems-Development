import pandas as pd
import json
from pathlib import Path
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

YELP_BUSINESS_FILE = Path("./yelp_dataset/yelp_academic_dataset_business.json")
YELP_REVIEW_FILE = Path("./yelp_dataset/yelp_academic_dataset_review.json")
YELP_USER_FILE = Path("./yelp_dataset/yelp_academic_dataset_user.json")

OUTPUT_DIR = Path("./motorev_yelp_tourism")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tourism categories
TOURISM_CATEGORIES = [
    "Restaurants", "Hotels & Travel", "Nightlife", "Arts & Entertainment",
    "Active Life", "Shopping", "Food", "Event Planning & Services",
    "Beauty & Spas", "Local Services", "Tours", "Museums", "Landmarks"
]

print("="*70)
print("EXTRACTING YELP DATA - MULTI-CITY TOURISM")
print("="*70)

# ============================================================================
# STEP 0: DISCOVER ACTUAL CITY NAMES
# ============================================================================

print("\n🔍 Scanning dataset to find actual city names...")
city_counter = {}

with open(YELP_BUSINESS_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 150000:  # Check first 150k records
            break
        if i % 50000 == 0:
            print(f"  Scanned {i:,} records...")
        
        try:
            business = json.loads(line)
            city = business.get('city', '').strip()
            if city:
                city_counter[city] = city_counter.get(city, 0) + 1
        except:
            continue

print("\n📊 Top 30 cities found:")
for city, count in sorted(city_counter.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"  {city}: {count:,}")

print("\n" + "="*70)
input("⏸️  Review the city list above. Press ENTER to continue or Ctrl+C to abort...")

# ============================================================================
# NOW SET YOUR TARGET CITIES (use exact names from above)
# ============================================================================

TARGET_CITIES = [
    "Philadelphia",      # Adjust based on what you see above
    "Tucson",
    "Tampa"# Adjust based on what you see above
    # Add more exact matches from the list above
]

print(f"\n✅ Target cities: {TARGET_CITIES}")

MAX_BUSINESSES = 9999999  # No limit for now
MAX_USERS = 9999999       # No limit for now
MAX_REVIEWS = 9999999      # Stop after 100k reviews for speed

# ============================================================================
# STEP 1: LOAD & FILTER BUSINESSES
# ============================================================================

print("\n📍 Loading businesses...")
businesses = []

with open(YELP_BUSINESS_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(f"  Processed {i:,} business records...")
        
        try:
            business = json.loads(line)
            
            # Filter by target cities
            if business.get('city') not in TARGET_CITIES:
                continue
                
            # Filter by tourism categories
            categories = business.get('categories', '')
            if not categories or not any(cat in categories for cat in TOURISM_CATEGORIES):
                continue
                
            businesses.append({
                'business_id': business['business_id'],
                'name': business['name'],
                'city': business['city'],
                'state': business.get('state', ''),
                'categories': categories,
                'stars': business.get('stars', 0),
                'review_count': business.get('review_count', 0)
            })
        except:
            continue

businesses = pd.DataFrame(businesses)

print(f"\n✓ Found {len(businesses):,} tourism businesses")
print(f"  Distribution by city:")
for city in TARGET_CITIES:
    count = len(businesses[businesses['city'] == city])
    print(f"    - {city}: {count:,}")

# Keep top businesses by review count
businesses = businesses.nlargest(min(MAX_BUSINESSES, len(businesses)), 'review_count')

print(f"\n✓ Keeping top {len(businesses):,} businesses by review count")

# ============================================================================
# STEP 2: LOAD REVIEWS (OPTIMIZED)
# ============================================================================

print("\n📝 Loading reviews (optimized for speed)...")

business_ids_set = set(businesses['business_id'])
reviews = []

with open(YELP_REVIEW_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(f"  Processed {i:,} reviews... (found {len(reviews):,} relevant)")
        
        # Stop early if we have enough
        #if len(reviews) >= MAX_REVIEWS:
        #    print(f"  ✓ Reached {MAX_REVIEWS:,} reviews, stopping for speed")
        #    break
        
        try:
            review = json.loads(line)
            
            if review.get('business_id') not in business_ids_set:
                continue
            
            reviews.append({
                'user_id': review['user_id'],
                'business_id': review['business_id'],
                'stars': review['stars']
            })
        except:
            continue

reviews = pd.DataFrame(reviews)

print(f"\n✓ Loaded {len(reviews):,} reviews")
print(f"  - Unique users: {reviews['user_id'].nunique():,}")
print(f"  - Unique businesses: {reviews['business_id'].nunique():,}")

# ============================================================================
# STEP 3: FILTER USERS BY RATING COUNT
# ============================================================================

print("\n👥 Filtering users...")

user_counts = reviews['user_id'].value_counts()

print(f"  User distribution:")
print(f"    - Users with 1-5 ratings: {(user_counts <= 5).sum():,}")
print(f"    - Users with 6-10 ratings: {((user_counts > 5) & (user_counts <= 10)).sum():,}")
print(f"    - Users with 11-20 ratings: {((user_counts > 10) & (user_counts <= 20)).sum():,}")
print(f"    - Users with 20+ ratings: {(user_counts > 20).sum():,}")

# Keep users with at least 10 ratings
MIN_RATINGS_PER_USER = 10
valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
reviews = reviews[reviews['user_id'].isin(valid_users)]

print(f"\n✓ Kept users with ≥{MIN_RATINGS_PER_USER} ratings: {reviews['user_id'].nunique():,}")

# Keep only top N users by rating count
MAX_FINAL_USERS = 3000
top_users = reviews['user_id'].value_counts().head(MAX_FINAL_USERS).index
reviews = reviews[reviews['user_id'].isin(top_users)]

print(f"✓ Limited to top {MAX_FINAL_USERS:,} users")

# ============================================================================
# STEP 4: CREATE MAPPINGS & SAVE
# ============================================================================

print("\n💾 Creating mappings and saving files...")

# Create user mapping
unique_users = sorted(reviews['user_id'].unique())
user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}

# Create business mapping (only businesses that have reviews)
reviewed_businesses = reviews['business_id'].unique()
businesses_filtered = businesses[businesses['business_id'].isin(reviewed_businesses)].copy()

# Map businesses to indices
unique_business_ids = sorted(businesses_filtered['business_id'].unique())
business_to_idx = {bid: idx for idx, bid in enumerate(unique_business_ids)}

# Map reviews to indices
reviews['user_idx'] = reviews['user_id'].map(user_to_idx)
reviews['business_idx'] = reviews['business_id'].map(business_to_idx)

# Save ratings.txt
ratings_output = reviews[['user_idx', 'business_idx', 'stars']]
ratings_output.to_csv(OUTPUT_DIR / "ratings.txt", sep='\t', header=False, index=False)
print(f"✓ Saved ratings: {len(ratings_output):,} ratings")

# Save users.csv
users_df = pd.DataFrame({
    'user_idx': range(len(unique_users)),
    'user_id': unique_users
})
users_df.to_csv(OUTPUT_DIR / "users.csv", index=False)
print(f"✓ Saved users: {len(users_df):,} users")

# ✅ FIXED: Save businesses.csv with ALL required columns
businesses_filtered['business_idx'] = businesses_filtered['business_id'].map(business_to_idx)

# Extract primary category
def get_primary_category(cat_str):
    if pd.isna(cat_str) or cat_str == '':
        return 'Restaurants'
    return str(cat_str).split(',')[0].strip()

businesses_filtered['primary_category'] = businesses_filtered['categories'].apply(get_primary_category)

# Select and save required columns
businesses_output = businesses_filtered[[
    'business_idx',
    'business_id', 
    'name',
    'city',
    'categories',
    'primary_category',
    'stars',
    'review_count'
]].copy()

# Add lat/lon if available, otherwise default to 0
if 'latitude' not in businesses_output.columns:
    businesses_output['latitude'] = 0.0
if 'longitude' not in businesses_output.columns:
    businesses_output['longitude'] = 0.0

businesses_output.to_csv(OUTPUT_DIR / "businesses.csv", index=False)
print(f"✓ Saved businesses: {len(businesses_output):,} businesses")

# Show what was saved
print(f"\n📋 Saved columns: {list(businesses_output.columns)}")


# ============================================================================
# SUMMARY
# ============================================================================
# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ EXTRACTION COMPLETE")
print("="*70)

# Calculate statistics
num_users = len(users_df)
num_businesses = len(businesses_output)
num_ratings = len(ratings_output)
avg_ratings_per_user = num_ratings / num_users
density_pct = 100 * num_ratings / (num_users * num_businesses)

print(f"\n📊 Final Dataset Statistics:")
print(f"  - Users: {num_users:,}")
print(f"  - Businesses: {num_businesses:,}")
print(f"  - Ratings: {num_ratings:,}")
print(f"  - Avg ratings/user: {avg_ratings_per_user:.1f}")
print(f"  - Density: {density_pct:.3f}%")

# City distribution
print(f"\n🌍 Business Distribution by City:")
for city in businesses_output['city'].value_counts().head(10).items():
    print(f"  - {city[0]}: {city[1]:,} businesses")

# Category distribution
print(f"\n📂 Top Categories:")
for cat in businesses_output['primary_category'].value_counts().head(10).items():
    print(f"  - {cat[0]}: {cat[1]:,} businesses")

# Rating distribution
print(f"\n⭐ Rating Distribution:")
for star in sorted(ratings_output['stars'].unique()):
    count = (ratings_output['stars'] == star).sum()
    pct = 100 * count / num_ratings
    print(f"  - {star:.1f} stars: {count:,} ({pct:.1f}%)")

print(f"\n📂 Output Directory: {OUTPUT_DIR.absolute()}")
print(f"   - ratings.txt ({num_ratings:,} rows)")
print(f"   - users.csv ({num_users:,} rows)")
print(f"   - businesses.csv ({num_businesses:,} rows)")

print(f"\n🚀 Next Step: Run Part 1 to generate domain model")
print(f"   python extractYelp.py")
