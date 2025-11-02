"""Helper functions for generating crop-specific advice and recommendations."""

def get_crop_specific_advice(crop_name):
    """Generate natural language advice specific to each crop."""
    crop_advice = {
        'rice': ("Rice thrives in your conditions. Maintain consistent water levels during the growing season "
                "and consider using disease-resistant varieties."),
        'wheat': ("Wheat adapts well to these conditions. Focus on proper seed spacing and timely "
                 "weed management for optimal results."),
        'maize': ("Maize/Corn shows good potential here. Ensure adequate spacing between plants "
                 "and consider companion planting with beans or squash."),
        'cotton': ("Cotton can flourish in these conditions. Implement proper pest monitoring "
                  "and maintain good air circulation between plants."),
        'sugarcane': ("Sugarcane growth looks promising. Maintain proper row spacing and "
                     "implement good drainage systems."),
        'tea': ("Tea can thrive here. Focus on canopy management and ensure good air flow "
               "between plants to prevent fungal issues."),
        'coffee': ("Coffee shows potential. Consider providing partial shade and maintain "
                  "organic matter in the soil through mulching."),
        'coconut': ("Coconut palms can do well. Ensure adequate spacing between trees and "
                   "maintain soil moisture through mulching."),
        'jute': ("Jute adaptation looks good. Focus on maintaining soil moisture and "
                "implementing proper weed management."),
        'groundnut': ("Groundnuts/Peanuts can thrive here. Practice crop rotation and maintain "
                    "loose, well-drained soil conditions.")
    }
    
    return crop_advice.get(crop_name.lower(), 
                          "This crop shows good potential in your conditions. "
                          "Monitor soil moisture and nutrient levels regularly.")

def get_fertilizer_recommendations(soil_params):
    """Generate fertilizer recommendations based on soil parameters."""
    recommendations = []
    
    # Nitrogen recommendations
    if soil_params.get('soil_nitrogen', 0) < 140:
        recommendations.append({
            'nutrient': 'Nitrogen',
            'status': 'Low',
            'fertilizers': ['Urea', 'Ammonium Sulfate', 'Calcium Nitrate'],
            'organic_options': ['Compost', 'Blood Meal', 'Fish Emulsion']
        })
    
    # Phosphorus recommendations
    if soil_params.get('soil_phosphorus', 0) < 10:
        recommendations.append({
            'nutrient': 'Phosphorus',
            'status': 'Low',
            'fertilizers': ['Triple Superphosphate', 'Rock Phosphate', 'DAP'],
            'organic_options': ['Bone Meal', 'Fish Bone Meal', 'Bat Guano']
        })
    
    # Potassium recommendations
    if soil_params.get('soil_potassium', 0) < 20:
        recommendations.append({
            'nutrient': 'Potassium',
            'status': 'Low',
            'fertilizers': ['Potassium Chloride', 'Potassium Sulfate', 'NPK Mix'],
            'organic_options': ['Wood Ash', 'Seaweed Extract', 'Banana Peels']
        })
    
    return recommendations