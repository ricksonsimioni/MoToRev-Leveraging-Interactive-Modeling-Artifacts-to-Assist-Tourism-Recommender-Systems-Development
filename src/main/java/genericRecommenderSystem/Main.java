package genericRecommenderSystem;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EEnumLiteral;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.epsilon.emc.emf.EmfModel;
import org.eclipse.epsilon.eol.EolModule;
import org.eclipse.epsilon.eol.IEolModule;
import org.eclipse.epsilon.eol.exceptions.models.EolModelLoadingException;
import org.eclipse.epsilon.eol.models.IModel;

public class Main {
    private static final Random random = new Random();

    public static void main(String[] args) {
    	
    	
        try {
            ResourceSet resourceSetTRS = new ResourceSetImpl();
            resourceSetTRS.getResourceFactoryRegistry().getExtensionToFactoryMap().put("ecore", new EcoreResourceFactoryImpl());
            Resource ecoreResourceTRS = resourceSetTRS.getResource(URI.createFileURI("src/main/Models/recommendersystemGeneric.ecore"), true);

            ResourceSet resourceSetDomain = new ResourceSetImpl();
            resourceSetDomain.getResourceFactoryRegistry().getExtensionToFactoryMap().put("ecore", new EcoreResourceFactoryImpl());
            Resource ecoreResourceDomain = resourceSetDomain.getResource(URI.createFileURI("src/main/Models/domain.ecore"), true);
            //to test the movie domain, uncomment the next line and comment the previous one.
            //Resource ecoreResourceDomain = resourceSetDomain.getResource(URI.createFileURI("src/main/Models/movieDomain.ecore"), true);

            EPackage ePackageRS = (EPackage) ecoreResourceTRS.getContents().get(0);
            EPackage ePackageDomain = (EPackage) ecoreResourceDomain.getContents().get(0);

            EPackage.Registry.INSTANCE.put(ePackageRS.getNsURI(), ePackageRS);
            EPackage.Registry.INSTANCE.put(ePackageDomain.getNsURI(), ePackageDomain);


            IEolModule module = new EolModule();
            module.parse(new File("src/main/Models/EOL_scripts/dataExtraction.eol"));
            if (module.getParseProblems().size() > 0) {
                System.err.println("Parse problems occurred: " + module.getParseProblems());
            }


            try {
                List<IModel> models = new ArrayList<>();
                
                models.add(loadEmfModel("recommendersystemModel", new File("src/main/Models/recommendersystemGeneric.model").getAbsolutePath(), "http://org.rs", true, false));
                models.add(loadEmfModel("domain", new File("src/main/Models/GranParadiso.model").getAbsolutePath(), "http://org.rs.domain", true, false));            

                for (IModel model : models) {
                    module.getContext().getModelRepository().addModel(model);
                }
                
                Object result = module.execute();

                for (IModel model : models) {
                    model.dispose();
                }

                EEnum categoryEnum = (EEnum) ePackageDomain.getEClassifier("Category");
                List<String> categoryValues = new ArrayList<>();

                if (categoryEnum != null && categoryEnum.getELiterals() != null) {
                    for (EEnumLiteral literal : categoryEnum.getELiterals()) {
                        categoryValues.add(literal.getName());
                    }
                }

                if (result instanceof Map) {
                    Map<?, ?> resultMap = (Map<?, ?>) result;

                    Object ratingsObj = resultMap.get("ratingsData");
                    FastByIDMap<PreferenceArray> mahoutData = null;
                    
                    if (ratingsObj instanceof List) {
                        mahoutData = processData(extractRatingsData((List<?>) ratingsObj));
                        //System.out.println("Mahout data: " + mahoutData);
                        //System.out.println("Ratings Object: " + ratingsObj);
                        //System.out.println("Size of Ratings Object: " + ((List<?>) ratingsObj).size());

                    }
                    Map<Integer, Map<String, Object>> itemData = extractItemData(resultMap);

                    
                    if (mahoutData != null) {
                        DataModel model = new GenericDataModel(mahoutData);
                        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
                        UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);
                        GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

                        try (FileWriter csvWriter = new FileWriter("recommendationsParkParadiso.csv");
                          	 FileWriter modelWriter = new FileWriter("src/Main/Models/output/recommendationsParkParadiso.model")) {

                        //try (FileWriter csvWriter = new FileWriter("recommendationsMovieLens.csv");
                             //FileWriter modelWriter = new FileWriter("src/Main/Models/output/recommendationsMovieLens.model")) {

                                csvWriter.append("user_id,item_id,rating,category\n");
                                
                                // Write XML Header
                                modelWriter.append("<?xml version=\"1.0\" encoding=\"ASCII\"?>\n");
                                modelWriter.append("<recommendations:UserItemMatrix xmi:version=\"2.0\" xmlns:xmi=\"http://www.omg.org/XMI\" xmlns:recommendations=\"http://org.rs.recommendations\" xmi:id=\"_BDihILMeEe-cHalY6VHacQ\">\n");
                                //modelWriter.append("<recommendations:UserItemMatrix xmi:version=\"2.0\" xmlns:xmi=\"http://www.omg.org/XMI\" xmlns:recommendations=\"http://org.rs.recommendations\" xmi:id=\"_BVihILMeEe-cHalY6VHacQ\">\n");

                            for (LongPrimitiveIterator users = model.getUserIDs(); users.hasNext();) {
                                long userId = users.nextLong();

                                // Aggregate recommendations across categories
                                Map<Long, List<RecommendedItem>> aggregatedRecommendations = new HashMap<>();
                                for (String favoriteCategory : categoryValues) {
                                    List<RecommendedItem> recommendations = recommender.recommend(userId, 10);
                                    List<RecommendedItem> filteredRecommendations = rerankRecommendations(recommendations, favoriteCategory, itemData, userId);

                                    aggregatedRecommendations.computeIfAbsent(userId, k -> new ArrayList<>()).addAll(filteredRecommendations);
                                }

                                // Final recommendations: remove duplicates, rank, and limit
                                List<RecommendedItem> finalRecommendations = aggregatedRecommendations.get(userId).stream()
                                    .distinct() // Remove duplicate items
                                    .sorted(Comparator.comparingDouble(RecommendedItem::getValue).reversed()) // Re-rank across categories
                                    .limit(10) // Limit to 10 items
                                    .collect(Collectors.toList());

                                // Write final recommendations to the CSV
                                for (RecommendedItem recommendation : finalRecommendations) {
                                    int itemId = (int) recommendation.getItemID();
                                    float rating = Math.round(recommendation.getValue() * 100.0f) / 100.0f;
                                    String category = (itemData.containsKey(itemId) && itemData.get(itemId).containsKey("category"))
                                            ? itemData.get(itemId).get("category").toString()
                                            : "Unknown";

                                    csvWriter.append(userId + "," + itemId + "," + rating + "," + category + "\n");
                                    	//System.out.println(" Item " + itemId + " recommended for user " + userId + " with rating: " + rating + ", based on category: " + category);
                                 // Write to XML
                                    String randomId = generateRandomId();
                                    modelWriter.append("  <rows xmi:id=\"" + randomId + "\" userId=\"" + userId + "\" itemId=\"" + itemId + "\" rating=\"" + rating + "\" category=\"" + category + "\"/>\n");                                
                                }
                            }
                            
                            modelWriter.append("</recommendations:UserItemMatrix>\n");
                            csvWriter.flush();
                            modelWriter.flush();
                            System.out.println("Recommendations saved to recommendationsPark.csv/model");

                        } catch (IOException e) {
                            System.err.println("Error writing to CSV: " + e.getMessage());
                        }

                    }
                }
            } catch (EolModelLoadingException e) {
                e.printStackTrace();
                System.err.println("Error loading model: " + e.getMessage());
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Error executing EOL script: " + e.getMessage());
        }
    }
    
    
    private static String generateRandomId() {
    	 StringBuilder id = new StringBuilder("_");
         for (int i = 0; i < 10; i++) {
             id.append((char) ('A' + random.nextInt(26))); // Random uppercase letter
         }
         id.append("-");
         id.append("LMfEe-cHalY6VHacQ"); // Fixed suffix
         return id.toString();
	}


    private static List<RecommendedItem> rerankRecommendations(List<RecommendedItem> recommendations, 
            String favoriteCategory, 
            Map<Integer, Map<String, Object>> itemData,
            long userId) {  // Added userId parameter
        
        // Debug: Print the favorite category for this user
        System.out.println("Debug: User " + userId + "'s favorite category is: " + favoriteCategory);
        
        // Original filtering and sorting logic
        return recommendations.stream()
            .filter(rec -> {
                Map<String, Object> itemDetails = itemData.get((int) rec.getItemID());
                if (itemDetails != null) {
                    Object categoryObj = itemDetails.get("category");
                    return categoryObj != null && favoriteCategory.equalsIgnoreCase(categoryObj.toString().trim());
                }
                return false;
            })
            .sorted(Comparator.comparingDouble(RecommendedItem::getValue).reversed())
            .collect(Collectors.toList());
    }
    
    public static Map<Integer, Map<String, Object>> extractItemData(Map<?, ?> resultMap) {
        Map<Integer, Map<String, Object>> itemData = new HashMap<>();
        
        if (resultMap instanceof Map) {
            Object itemDataObj = resultMap.get("itemData");
            if (itemDataObj instanceof Map) {
                Map<?, ?> rawItemData = (Map<?, ?>) itemDataObj;
                if (!rawItemData.isEmpty()) {
                    rawItemData.forEach((key, value) -> {
                        if (value instanceof Map) {
                            Map<?, ?> details = (Map<?, ?>) value;
                            if (details.containsKey("itemId")) {
                                Object itemIdObj = details.get("itemId");
                                if (itemIdObj instanceof Integer) {
                                    Integer itemId = (Integer) itemIdObj;
                                    Map<String, Object> itemDetails = new HashMap<>();
                                    details.forEach((detailKey, detailValue) -> {
                                        if (detailKey instanceof String) {
                                            itemDetails.put((String) detailKey, detailValue);
                                        }
                                    });
                                    itemData.put(itemId, itemDetails);
                                } else {
                                    System.out.println("itemId is not an integer.");
                                }
                            } else {
                                System.out.println("itemId is missing.");
                            }
                        }
                    });
                } else {
                    System.out.println("Item Data is empty.");
                }
            } else {
                System.out.println("Item Data object is not a map.");
            }
        } else {
            System.out.println("Result object is not a map.");
        }
        
        return itemData;
    }


    private static Map<Long, List<Object[]>> extractRatingsData(List<?> ratingsList) {
        Map<Long, List<Object[]>> ratingsData = new HashMap<>();
        for (Object obj : ratingsList) {
            if (obj instanceof Map) {
                Map<?, ?> rating = (Map<?, ?>) obj;
                try {
                    Long userId = Long.parseLong(rating.get("userId").toString());
                    Long itemId = Long.parseLong(rating.get("itemId").toString());
                    float value = Float.parseFloat(rating.get("rating").toString());

                    ratingsData.computeIfAbsent(userId, k -> new ArrayList<>()).add(new Object[]{itemId, value});
                } catch (NumberFormatException e) {
                    System.err.println("Error parsing rating data: " + e.getMessage());
                }
            }
        }
        return ratingsData;
    }


    private static FastByIDMap<PreferenceArray> processData(Map<Long, List<Object[]>> ratingsData) {
        FastByIDMap<PreferenceArray> userData = new FastByIDMap<>();

        ratingsData.forEach((userId, preferences) -> {
            PreferenceArray prefsArray = new GenericUserPreferenceArray(preferences.size());
            for (int i = 0; i < preferences.size(); i++) {
                Object[] prefDetails = preferences.get(i);
                long itemId = (long) prefDetails[0];
                float rating = (float) prefDetails[1];
                prefsArray.setUserID(i, userId);
                prefsArray.setItemID(i, itemId);
                prefsArray.setValue(i, rating);
            }
            userData.put(userId, prefsArray);
        });

        return userData;
    }

    public static EmfModel loadEmfModel(String name, String modelPath, String metamodelUri, boolean readOnLoad, boolean storeOnDisposal) throws Exception {
        EmfModel model = new EmfModel();
        model.setName(name);
        model.setMetamodelUri(metamodelUri);
        model.setModelFile(modelPath);
        model.setReadOnLoad(readOnLoad);
        model.setStoredOnDisposal(storeOnDisposal);
        model.setExpand(true);
        model.load();
        return model;
    }
}