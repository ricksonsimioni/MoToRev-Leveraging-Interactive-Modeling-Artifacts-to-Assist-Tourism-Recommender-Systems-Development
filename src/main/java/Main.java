import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

import com.google.common.collect.BiMap;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.XMLResource;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;

import org.eclipse.epsilon.eol.IEolModule;
import org.eclipse.epsilon.eol.EolModule;
import org.eclipse.epsilon.eol.models.IModel;
import org.eclipse.epsilon.emc.emf.EmfModel;

import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.eval.EvalContext;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.ranking.PrecisionEvaluator;
import net.librec.eval.ranking.RecallEvaluator;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.ranking.GBPRRecommender;
import net.librec.recommender.item.KeyValue;
import net.librec.recommender.item.RecommendedList;

public class Main {

    static final String LIBREC_INPUT_FILENAME = "librec_input.txt";
    static final String LIBREC_ITEM_CATEGORY_FILENAME = "librec_item_categories.txt";

    private static RecommendedList lastRecommendedList;
    private static TextDataModel lastDataModel;
    private static double lastNdcg = 0.0;
    private static double lastPrecision = 0.0;
    private static double lastRecall = 0.0;
    private static double lastF1 = 0.0;

    // === Helper classes ===

    static class EOLData {
        List<Map<String, Object>> ratingsData;
        List<Map<String, Object>> itemData;

        String rsType;   // "CF", "CB", "HYBRID", "UNKNOWN"
        boolean hasCF;
        boolean hasCB;

        EOLData(List<Map<String, Object>> ratings,
                List<Map<String, Object>> categories,
                String rsType,
                boolean hasCF,
                boolean hasCB) {

            this.ratingsData = (ratings != null) ? ratings : new ArrayList<>();
            this.itemData = (categories != null) ? categories : new ArrayList<>();
            this.rsType = (rsType != null) ? rsType : "UNKNOWN";
            this.hasCF = hasCF;
            this.hasCB = hasCB;
        }
    }


    static class IdMappings {
        Map<String, Integer> userMap;
        Map<String, Integer> itemMap;

        IdMappings(Map<String, Integer> u, Map<String, Integer> i) {
            this.userMap = u;
            this.itemMap = i;
        }
    }

    static class GridSearchResult {
        int factors;
        double learningRate;
        int groupSize;
        double regularization;
        double[] ndcgScores;
        double[] precisionScores;
        double[] recallScores;
        double avgNDCG;
        double medianNDCG;
        double stdNDCG;
        
        GridSearchResult(int f, double lr, int gs, double reg) {
            this.factors = f;
            this.learningRate = lr;
            this.groupSize = gs;
            this.regularization = reg;
        }
        
        void computeStats(int runs) {
            Arrays.sort(ndcgScores);
            Arrays.sort(precisionScores);
            Arrays.sort(recallScores);
            
            avgNDCG = Arrays.stream(ndcgScores).average().orElse(0.0);
            medianNDCG = ndcgScores[runs / 2];
            
            double variance = 0.0;
            for (double score : ndcgScores) {
                variance += Math.pow(score - avgNDCG, 2);
            }
            stdNDCG = Math.sqrt(variance / runs);
        }
        
        @Override
        public String toString() {
            return String.format("factors=%d, lr=%.3f, groupSize=%d, reg=%.4f | " +
                    "NDCG: avg=%.4f, median=%.4f, std=%.4f",
                    factors, learningRate, groupSize, regularization,
                    avgNDCG, medianNDCG, stdNDCG);
        }
    }

    // === Getters for dashboard ===

    public static RecommendedList getLastRecommendedList() {
        return lastRecommendedList;
    }

    public static TextDataModel getLastDataModel() {
        return lastDataModel;
    }

    public static double getLastNdcg() {
        return lastNdcg;
    }

    public static double getLastPrecision() {
        return lastPrecision;
    }

    public static double getLastRecall() {
        return lastRecall;
    }

    public static double getLastF1() {
        return lastF1;
    }

    // === Memory logging ===

    public static void logMemoryUsage(String phase) {
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        System.out.printf("%s - Memory: %d MB used, %d MB free, %d MB total, %d MB max%n",
                phase,
                usedMemory / (1024 * 1024),
                runtime.freeMemory() / (1024 * 1024),
                runtime.totalMemory() / (1024 * 1024),
                maxMemory / (1024 * 1024));
    }

    // === Config loader ===

    private static Properties loadConfigFromArgs() throws IOException {
        Properties props = new Properties();
        String configFile = System.getProperty("config.file");

        if (configFile != null && new File(configFile).exists()) {
            try (InputStream in = new FileInputStream(configFile)) {
                props.load(in);
                System.out.println("Loaded configuration from: " + configFile);
            }
        } else {
            try (InputStream in = Main.class.getClassLoader().getResourceAsStream("config.properties")) {
                if (in == null) {
                    throw new FileNotFoundException("config.properties not found in classpath and no config.file specified");
                }
                props.load(in);
                System.out.println("Loaded configuration from classpath");
            }
        }
        return props;
    }

    // === Main ===

    public static void main(String[] args) {
        try {
            logMemoryUsage("Start");

            Properties props = loadConfigFromArgs();
            double cfWeight = Double.parseDouble(props.getProperty("hybrid.weight", "0.9"));

            // 1. EOL extraction
            logMemoryUsage("Before EOL extraction");
            EOLData eolData = extractRatingsFromEOL(props);
            logMemoryUsage("After EOL extraction");
            
            System.out.printf(
            	    "Configured RS type from model: %s (hasCF=%s, hasCB=%s)%n",
            	    eolData.rsType, eolData.hasCF, eolData.hasCB
            	);
            
            if ("CF".equalsIgnoreCase(eolData.rsType)) {
                // Pure collaborative filtering
                cfWeight = 1.0;
                System.out.println("Model specifies CollaborativeFiltering; forcing pure CF (cfWeight=1.0).");
            } else if ("CB".equalsIgnoreCase(eolData.rsType)) {
                // Pure content-based
                cfWeight = 0.0;
                System.out.println("Model specifies ContentBased; forcing pure CB (cfWeight=0.0).");
            } else if ("HYBRID".equalsIgnoreCase(eolData.rsType)) {
                // Keep configured hybrid weight (e.g., 0.9) or override via property
                System.out.println("Model specifies HybridBased; using hybrid CF+CB (cfWeight=" + cfWeight + ").");
            } else {
                // Fallback to existing behaviour (data-driven inference)
                System.out.println("RS type UNKNOWN from model; falling back to property-based hybrid.weight.");
            }

            if (eolData.ratingsData.isEmpty()) {
                System.out.println("No rating data available from the EOL script. Exiting.");
                return;
            }
            
            if (eolData.itemData.isEmpty()) {
                System.out.println("Warning: No item category data found from EOL script. Hybrid recommendation will be purely collaborative.");
                cfWeight = 1.0;
            }

            Set<String> uniqueItemsFromRatings = eolData.ratingsData.stream()
                    .map(m -> m.get("itemId").toString())
                    .collect(Collectors.toSet());
            Set<String> uniqueItemsFromCategories = eolData.itemData.stream()
                    .map(m -> m.get("itemId").toString())
                    .collect(Collectors.toSet());

            System.out.printf("DEBUG: Extracted %d ratings from EOL. Unique Users: %d, Unique Items (Ratings): %d%n",
                    eolData.ratingsData.size(),
                    eolData.ratingsData.stream().map(m -> m.get("userId").toString()).distinct().count(),
                    uniqueItemsFromRatings.size());

            System.out.printf("DEBUG: Extracted %d item-category entries from EOL. Unique Categories: %d, Unique Items (Categories): %d%n",
                    eolData.itemData.size(),
                    eolData.itemData.stream()
                            .map(m -> m.get("category") == null ? "" : m.get("category").toString())
                            .filter(s -> !s.isEmpty())
                            .distinct()
                            .count(),
                    uniqueItemsFromCategories.size());

            // Build itemId -> categories lookup
            Map<String, Set<String>> itemCategoryLookup = new HashMap<>();
            for (Map<String, Object> itemCat : eolData.itemData) {
                Object itemIdObj = itemCat.get("itemId");
                Object categoryObj = itemCat.get("category");
                if (itemIdObj == null || categoryObj == null) continue;
                String itemId = itemIdObj.toString();
                String category = categoryObj.toString();
                itemCategoryLookup.computeIfAbsent(itemId, k -> new HashSet<>()).add(category);
            }

            // 2. Write LibRec input files
            Path tmpDir = Paths.get(props.getProperty("output.tmp")).toAbsolutePath();
            Files.createDirectories(tmpDir);

            Path librecDataFile = tmpDir.resolve(LIBREC_INPUT_FILENAME);
            IdMappings mappings = writeLibrecInputFile(librecDataFile.toFile(), eolData.ratingsData);
            checkLibrecFileContiguity(librecDataFile.toFile());

            Path librecItemCategoryFile = tmpDir.resolve(LIBREC_ITEM_CATEGORY_FILENAME);
            writeLibrecItemCategoryFile(librecItemCategoryFile.toFile(), eolData.itemData, mappings.itemMap);

            eolData = null;
            System.gc();
            logMemoryUsage("After clearing EOL data");

            // 3. CF configuration + data model
            Configuration cfConf = prepareCfConfiguration(tmpDir, props);
            TextDataModel cfDataModel = new TextDataModel(cfConf);
            cfDataModel.buildDataModel();

            System.out.printf("DEBUG: CF Data Model Built. Users: %d, Items: %d, Ratings: %d%n",
                    cfDataModel.getUserMappingData().size(),
                    cfDataModel.getItemMappingData().size(),
                    cfDataModel.getTrainDataSet().size() + cfDataModel.getTestDataSet().size());

            // Check if grid search is enabled
            boolean gridSearchEnabled = Boolean.parseBoolean(System.getProperty("grid.search.enabled", 
                    props.getProperty("grid.search.enabled", "false")));
			int runsPerConfig = Integer.parseInt(System.getProperty("grid.search.runs",
			                    props.getProperty("grid.search.runs", "10")));
			
			if (gridSearchEnabled) {
			performGridSearch(tmpDir, props, cfDataModel, runsPerConfig);
			return;
			}

            // 4. CF recommender - GBPR
            System.out.println("=== Building CF Recommender (GBPR) ===");
            RecommenderContext cfContext = new RecommenderContext(cfConf, cfDataModel);

            Recommender cfRecommender = new GBPRRecommender();
            cfRecommender.setContext(cfContext);
            cfRecommender.train(cfContext);
            RecommendedList cfRecList = cfRecommender.recommendRank();
            System.out.printf("DEBUG: CF RecList size (Users with recommendations): %d / %d%n",
                    cfRecList.size(), cfDataModel.getUserMappingData().size());
            System.out.println("CF Recommender (GBPR) built.");

            // 5. CB recommender (TF-IDF weighted categories)
            System.out.println("=== Building CB Scores (TF-IDF Category Weighting) ===");
            RecommendedList cbRecList = new RecommendedList(cfDataModel.getUserMappingData().size());

            if (cfWeight < 1.0 && !itemCategoryLookup.isEmpty()) {
                BiMap<String, Integer> userMapping = cfDataModel.getUserMappingData();
                BiMap<String, Integer> itemMapping = cfDataModel.getItemMappingData();
                SequentialAccessSparseMatrix trainMatrix = (SequentialAccessSparseMatrix) cfDataModel.getTrainDataSet();

                // Calculate IDF for each category
                Map<String, Double> categoryIDF = new HashMap<>();
                int totalItems = itemMapping.size();
                for (String category : itemCategoryLookup.values().stream()
                        .flatMap(Set::stream).collect(Collectors.toSet())) {
                    long itemsWithCategory = itemCategoryLookup.values().stream()
                            .filter(cats -> cats.contains(category)).count();
                    double idf = Math.log((double) totalItems / (itemsWithCategory + 1));
                    categoryIDF.put(category, idf);
                }

                for (int userIdx = 0; userIdx < userMapping.size(); userIdx++) {
                    // Build user profile with TF-IDF
                    Map<String, Double> userProfile = new HashMap<>();
                    net.librec.math.structure.SequentialSparseVector userRatings = trainMatrix.row(userIdx);

                    for (int i = 0; i < userRatings.size(); i++) {
                        int itemIdx = userRatings.getIndexAtPosition(i);
                        double rating = userRatings.get(itemIdx);
                        String itemId = itemMapping.inverse().get(itemIdx);
                        Set<String> cats = itemCategoryLookup.get(itemId);
                        if (cats != null) {
                            for (String cat : cats) {
                                double tf = 1.0 / cats.size();
                                double weight = tf * categoryIDF.getOrDefault(cat, 1.0) * (rating / 5.0);
                                userProfile.merge(cat, weight, Double::sum);
                            }
                        }
                    }

                    // Score items by cosine similarity with user profile
                    List<KeyValue<Integer, Double>> cbScores = new ArrayList<>();
                    double userNorm = Math.sqrt(userProfile.values().stream()
                            .mapToDouble(v -> v * v).sum());
                    
                    if (userNorm > 0) {
                        for (Map.Entry<String, Integer> itemEntry : itemMapping.entrySet()) {
                            String itemId = itemEntry.getKey();
                            int itemIdx = itemEntry.getValue();
                            
                            Set<String> itemCats = itemCategoryLookup.get(itemId);
                            if (itemCats == null || itemCats.isEmpty()) continue;
                            
                            // Cosine similarity
                            double dotProduct = 0.0;
                            double itemNorm = 0.0;
                            for (String cat : itemCats) {
                                double itemWeight = categoryIDF.getOrDefault(cat, 1.0) / itemCats.size();
                                dotProduct += userProfile.getOrDefault(cat, 0.0) * itemWeight;
                                itemNorm += itemWeight * itemWeight;
                            }
                            itemNorm = Math.sqrt(itemNorm);
                            
                            if (itemNorm > 0) {
                                double score = dotProduct / (userNorm * itemNorm);
                                if (score > 0) {
                                    cbScores.add(new KeyValue<>(itemIdx, score));
                                }
                            }
                        }
                    }
                    
                    cbScores.sort((kv1, kv2) -> kv2.getValue().compareTo(kv1.getValue()));
                    cbRecList.addList(new ArrayList<>(cbScores));
                }
                System.out.println("CB Scores computed using TF-IDF weighting.");
            } else {
                System.out.println("Skipping CB (no categories or hybrid weight is 1.0).");
                for (int i = 0; i < cfDataModel.getUserMappingData().size(); i++) {
                    cbRecList.addList(new ArrayList<>());
                }
            }

            // 6. Hybrid
            System.out.println("=== Combining Recommendations (CF Weight: " + cfWeight + ") ===");
            RecommendedList hybridRecList = combineRecommendations(cfRecList, cbRecList, cfWeight, cfDataModel);
            System.out.printf("DEBUG: Hybrid RecList size (Users with recommendations): %d / %d%n",
                    hybridRecList.size(), cfDataModel.getUserMappingData().size());

            // 7. Debug
            debugTrainTestSplit(cfDataModel);
            debugContext(cfContext, cfRecommender, cfDataModel);

            // 8. Evaluation
            evaluateRecommendations(cfConf, cfRecommender, cfDataModel, cfContext, hybridRecList, props);

            // 9. Save
            saveRecommendationsCSV("recommendationsYelp.csv", hybridRecList, cfDataModel, itemCategoryLookup);
            saveRecommendationsModel("recommendationsYelp.model", hybridRecList, cfDataModel, itemCategoryLookup);

            lastRecommendedList = hybridRecList;
            lastDataModel = cfDataModel;

            logMemoryUsage("End");
            System.out.println("Done: hybrid recommendations saved (CSV + .model).");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // === Grid Search ===

    public static void performGridSearch(Path tmpDir, Properties props, 
                                         TextDataModel cfDataModel,
                                         int runsPerConfig) throws Exception {
        int topN = Integer.parseInt(props.getProperty("rec.topN", "10"));
        System.out.println("\n=== Starting Grid Search ===");
        System.out.println("Runs per configuration: " + runsPerConfig);
        
        // Parameter grid
        int[] factors = {64, 100, 128};
        double[] lrs = {0.01, 0.05, 0.1};
        int[] groupSizes = {3, 5, 10};
        double[] regularizations = {0.001, 0.01, 0.1};
        
        List<GridSearchResult> allResults = new ArrayList<>();
        int totalConfigs = factors.length * lrs.length * groupSizes.length * regularizations.length;
        int currentConfig = 0;
        
        long startTime = System.currentTimeMillis();
        
        // Try all combinations
        for (int f : factors) {
            for (double lr : lrs) {
                for (int gs : groupSizes) {
                    for (double reg : regularizations) {
                        currentConfig++;
                        System.out.printf("\n[%d/%d] Testing: factors=%d, lr=%.3f, groupSize=%d, reg=%.4f\n",
                                currentConfig, totalConfigs, f, lr, gs, reg);
                        
                        GridSearchResult result = new GridSearchResult(f, lr, gs, reg);
                        result.ndcgScores = new double[runsPerConfig];
                        result.precisionScores = new double[runsPerConfig];
                        result.recallScores = new double[runsPerConfig];
                        
                     // Run multiple times for statistics
                        for (int run = 0; run < runsPerConfig; run++) {
                            try {
                                Configuration conf = prepareCfConfiguration(tmpDir, props);
                                conf.setInt("rec.factor.number", f);
                                conf.setDouble("rec.iterator.learnrate", lr);
                                conf.setInt("rec.recommender.group.size", gs);
                                conf.setDouble("rec.user.regularization", reg);
                                conf.setDouble("rec.item.regularization", reg);
                                conf.setDouble("rec.group.regularization", reg);
                                
                                // Train recommender
                                RecommenderContext context = new RecommenderContext(conf, cfDataModel);
                                Recommender recommender = new GBPRRecommender();
                                recommender.setContext(context);
                                recommender.train(context);
                                RecommendedList recList = recommender.recommendRank();
                                
                                // Evaluate
                                EvalContext evalContext = new EvalContext(conf, recommender, 
                                        cfDataModel.getTestDataSet());
                                evalContext.setRecommendedList(recList);
                                
                                RecommenderEvaluator ndcgEval = new NormalizedDCGEvaluator();
                                ndcgEval.setTopN(topN);
                                result.ndcgScores[run] = ndcgEval.evaluate(evalContext);
                                
                                RecommenderEvaluator precisionEval = new PrecisionEvaluator();
                                precisionEval.setTopN(topN);
                                result.precisionScores[run] = precisionEval.evaluate(evalContext);
                                
                                RecommenderEvaluator recallEval = new RecallEvaluator();
                                recallEval.setTopN(topN);
                                result.recallScores[run] = recallEval.evaluate(evalContext);
                                
                                if ((run + 1) % 5 == 0) {
                                    System.out.printf("  Run %d/%d completed\n", run + 1, runsPerConfig);
                                }
                            } catch (Exception e) {
                                // Configuration failed (divergence, NaN, etc.)
                                System.out.printf("  Run %d/%d FAILED: %s\n", run + 1, runsPerConfig, 
                                        e.getMessage().split("\n")[0]);
                                // Mark as failed with score 0
                                result.ndcgScores[run] = 0.0;
                                result.precisionScores[run] = 0.0;
                                result.recallScores[run] = 0.0;
                            }
                        }
                     
                        

                        result.computeStats(runsPerConfig);
                        allResults.add(result);

                        // Only print result if it succeeded (avg > 0)
                        if (result.avgNDCG > 0) {
                            System.out.println("  Result: " + result);
                        } else {
                            System.out.println("  Result: CONFIGURATION FAILED (unstable parameters)");
                        }
   }
                }
            }
        }
        
        // Filter out failed configurations (avgNDCG = 0)
        allResults = allResults.stream()
                .filter(r -> r.avgNDCG > 0.0001)
                .collect(Collectors.toList());
        
        long endTime = System.currentTimeMillis();
        double totalMinutes = (endTime - startTime) / 60000.0;
        
        // Sort by average NDCG
        allResults.sort((a, b) -> Double.compare(b.avgNDCG, a.avgNDCG));
        
        // Save results to CSV
        saveGridSearchResults(allResults, "grid_search_results.csv");
        
        // Print summary
        System.out.println("\n=== Grid Search Complete ===");
        System.out.printf("Total time: %.2f minutes\n", totalMinutes);
        System.out.printf("Successful configurations: %d / %d\n", allResults.size(), totalConfigs);
        System.out.println("\nTop 10 configurations by average NDCG:");
        for (int i = 0; i < Math.min(10, allResults.size()); i++) {
            System.out.printf("%d. %s\n", i + 1, allResults.get(i));
        }
        
        GridSearchResult best = allResults.get(0);
        System.out.println("\n=== BEST CONFIGURATION ===");
        System.out.println(best);
        System.out.printf("Precision@10: avg=%.4f, median=%.4f\n", 
                Arrays.stream(best.precisionScores).average().orElse(0.0),
                best.precisionScores[runsPerConfig / 2]);
        System.out.printf("Recall@10: avg=%.4f, median=%.4f\n",
                Arrays.stream(best.recallScores).average().orElse(0.0),
                best.recallScores[runsPerConfig / 2]);
    }

    private static void saveGridSearchResults(List<GridSearchResult> results, String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("factors,learningRate,groupSize,regularization," +
                    "avgNDCG,medianNDCG,stdNDCG,minNDCG,maxNDCG," +
                    "avgPrecision,avgRecall\n");
            
            for (GridSearchResult r : results) {
                double avgPrecision = Arrays.stream(r.precisionScores).average().orElse(0.0);
                double avgRecall = Arrays.stream(r.recallScores).average().orElse(0.0);
                double minNDCG = Arrays.stream(r.ndcgScores).min().orElse(0.0);
                double maxNDCG = Arrays.stream(r.ndcgScores).max().orElse(0.0);
                
                writer.write(String.format("%d,%.4f,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        r.factors, r.learningRate, r.groupSize, r.regularization,
                        r.avgNDCG, r.medianNDCG, r.stdNDCG, minNDCG, maxNDCG,
                        avgPrecision, avgRecall));
            }
        }
        System.out.println("Grid search results saved to: " + filename);
    }

    // === EOL extraction ===

    public static EOLData extractRatingsFromEOL(Properties props) throws Exception {
        ResourceSet resourceSetTRS = new ResourceSetImpl();
        resourceSetTRS.getResourceFactoryRegistry().getExtensionToFactoryMap()
                .put("ecore", new EcoreResourceFactoryImpl());
        Resource ecoreResourceTRS = resourceSetTRS
                .getResource(URI.createFileURI(props.getProperty("ecore.recommender")), true);

        ResourceSet resourceSetDomain = new ResourceSetImpl();
        resourceSetDomain.getResourceFactoryRegistry().getExtensionToFactoryMap()
                .put("ecore", new EcoreResourceFactoryImpl());
        Resource ecoreResourceDomain = resourceSetDomain
                .getResource(URI.createFileURI(props.getProperty("ecore.domain")), true);

        EPackage ePackageRS = (EPackage) ecoreResourceTRS.getContents().get(0);
        EPackage ePackageDomain = (EPackage) ecoreResourceDomain.getContents().get(0);
        EPackage.Registry.INSTANCE.put(ePackageRS.getNsURI(), ePackageRS);
        EPackage.Registry.INSTANCE.put(ePackageDomain.getNsURI(), ePackageDomain);

        IEolModule module = new EolModule();
        module.parse(new File(props.getProperty("eol.script")));

        List<IModel> models = new ArrayList<>();
        try {
            models.add(loadEmfModel("recommendersystemModel",
                    new File(props.getProperty("model.recommender")).getAbsolutePath(),
                    ePackageRS.getNsURI(), true, false));
            models.add(loadEmfModel("domainModel",
                    new File(props.getProperty("model.domain")).getAbsolutePath(),
                    ePackageDomain.getNsURI(), true, false));

            for (IModel m : models) {
                module.getContext().getModelRepository().addModel(m);
            }

            Object result = module.execute();
            if (result == null) {
                System.out.println("EOL script result is null.");
            } else {
                System.out.println("EOL script result type: " + result.getClass().getName());
            }

            EOLData eolData = processEOLResult(result);
            if (eolData.itemData == null || eolData.itemData.isEmpty()) {
                System.out.println("Warning: Could not parse any item categories from the 'itemData' map in your EOL script.");
            }
            return eolData;
        } finally {
            for (IModel m : models) {
                try {
                    m.dispose();
                } catch (Exception e) {
                    System.err.println("Warning during model disposal: " + e.getMessage());
                }
            }
            models.clear();
            System.gc();
        }
    }

    @SuppressWarnings("unchecked")
    private static EOLData processEOLResult(Object result) {
        List<Map<String, Object>> ratings = new ArrayList<>();
        List<Map<String, Object>> items = new ArrayList<>();

        String rsType = "UNKNOWN";
        boolean hasCF = false;
        boolean hasCB = false;

        if (result == null) {
            return new EOLData(ratings, items, rsType, hasCF, hasCB);
        }

        if (result instanceof Map) {
            Map<Object, Object> tuple = (Map<Object, Object>) result;

            // --- existing parsing for ratingsData and itemData ---
            Object ratingsObj = tuple.get("ratingsData");
            if (ratingsObj instanceof Collection) {
                for (Object r : (Collection<?>) ratingsObj) {
                    if (r instanceof Map) {
                        Map<Object, Object> rm = (Map<Object, Object>) r;
                        Map<String, Object> row = new HashMap<>();
                        row.put("userId", rm.get("userId"));
                        row.put("itemId", rm.get("itemId"));
                        row.put("rating", rm.get("rating"));
                        ratings.add(row);
                    }
                }
            }

            Object itemDataObj = tuple.get("itemData");
            if (itemDataObj instanceof Map) {
                Map<Object, Object> itemMap = (Map<Object, Object>) itemDataObj;
                for (Map.Entry<Object, Object> e : itemMap.entrySet()) {
                    Object val = e.getValue();
                    if (val instanceof Map) {
                        Map<Object, Object> detail = (Map<Object, Object>) val;
                        Map<String, Object> row = new HashMap<>();
                        row.put("itemId", detail.get("itemId"));
                        Object categoryObj = detail.get("category");
                        String categoryName = null;
                        if (categoryObj != null) {
                            try {
                                categoryName =
                                    ((org.eclipse.emf.ecore.EObject) categoryObj).eGet(
                                        ((org.eclipse.emf.ecore.EObject) categoryObj)
                                            .eClass()
                                            .getEStructuralFeature("category")
                                    ).toString();
                            } catch (Exception x) {
                                categoryName = categoryObj.toString();
                            }
                        }
                        row.put("category", categoryName);
                        items.add(row);
                    }
                }
            }

            // --- NEW: read rsType, hasCF, hasCB if present ---
            Object rsTypeObj = tuple.get("rsType");
            if (rsTypeObj != null) {
                rsType = rsTypeObj.toString();
            }
            Object hasCFObj = tuple.get("hasCF");
            if (hasCFObj instanceof Boolean) {
                hasCF = (Boolean) hasCFObj;
            }
            Object hasCBObj = tuple.get("hasCB");
            if (hasCBObj instanceof Boolean) {
                hasCB = (Boolean) hasCBObj;
            }

            System.out.printf(
                "processEOLResult parsed %d ratings and %d item-category rows (rsType=%s, hasCF=%s, hasCB=%s)%n",
                ratings.size(), items.size(), rsType, hasCF, hasCB
            );

            return new EOLData(ratings, items, rsType, hasCF, hasCB);
        } else {
            System.out.println("Unexpected EOL result, expected Map-like Tuple.");
            return new EOLData(ratings, items, rsType, hasCF, hasCB);
        }
    }


    // === EMF model loader ===

    public static EmfModel loadEmfModel(String name,
                                        String modelPath,
                                        String metamodelUri,
                                        boolean readOnLoad,
                                        boolean storeOnDisposal) throws Exception {

        File modelFile = new File(modelPath);
        if (modelFile.isDirectory()) {
            EmfModel model = new EmfModel() {
                @Override
                protected ResourceSet createResourceSet() {
                    ResourceSet rs = super.createResourceSet();
                    Map<Object, Object> options = rs.getLoadOptions();

                    options.put(XMLResource.OPTION_DEFER_ATTACHMENT, Boolean.TRUE);
                    options.put(XMLResource.OPTION_DEFER_IDREF_RESOLUTION, Boolean.TRUE);
                    options.put(XMLResource.OPTION_USE_PARSER_POOL, new Object());
                    options.put(XMLResource.OPTION_USE_DEPRECATED_METHODS, Boolean.FALSE);
                    options.put(XMLResource.OPTION_USE_LEXICAL_HANDLER, Boolean.TRUE);
                    options.put(XMLResource.OPTION_DISABLE_NOTIFY, Boolean.TRUE);

                    rs.getResourceFactoryRegistry().getExtensionToFactoryMap()
                            .put("xmi", new EcoreResourceFactoryImpl());

                    File[] files = modelFile.listFiles((dir, name1) -> name1.toLowerCase().endsWith(".xmi"));
                    if (files != null) {
                        Arrays.sort(files, Comparator.comparing(File::getName));
                        for (File f : files) {
                            try {
                                rs.getResource(URI.createFileURI(f.getAbsolutePath()), true);
                            } catch (Exception ex) {
                                System.err.println("Warning loading fragment: " + f.getAbsolutePath() + " -> " + ex.getMessage());
                            }
                        }
                    }
                    return rs;
                }
            };

            model.setName(name);
            model.setMetamodelUri(metamodelUri);
            model.setModelFile(modelPath);
            model.setReadOnLoad(readOnLoad);
            model.setStoredOnDisposal(storeOnDisposal);
            model.setExpand(false);
            model.setCachingEnabled(false);
            model.load();
            return model;
        }

        EmfModel model = new EmfModel() {
            @Override
            protected ResourceSet createResourceSet() {
                ResourceSet rs = super.createResourceSet();
                Map<Object, Object> options = rs.getLoadOptions();

                options.put(XMLResource.OPTION_DEFER_ATTACHMENT, Boolean.TRUE);
                options.put(XMLResource.OPTION_DEFER_IDREF_RESOLUTION, Boolean.TRUE);
                options.put(XMLResource.OPTION_USE_PARSER_POOL, new Object());
                options.put(XMLResource.OPTION_USE_DEPRECATED_METHODS, Boolean.FALSE);
                options.put(XMLResource.OPTION_USE_LEXICAL_HANDLER, Boolean.TRUE);
                options.put(XMLResource.OPTION_DISABLE_NOTIFY, Boolean.TRUE);

                return rs;
            }
        };

        model.setName(name);
        model.setMetamodelUri(metamodelUri);
        model.setModelFile(modelPath);
        model.setReadOnLoad(readOnLoad);
        model.setStoredOnDisposal(storeOnDisposal);
        model.setExpand(false);
        model.setCachingEnabled(false);
        model.load();
        return model;
    }

    // === LibRec data writing ===

    public static IdMappings writeLibrecInputFile(File outFile, List<Map<String, Object>> ratingsList) throws IOException {
        Map<String, Integer> userMap = new HashMap<>();
        Map<String, Integer> itemMap = new HashMap<>();
        int uCounter = 0, iCounter = 0;

        for (Map<String, Object> r : ratingsList) {
            String uStr = r.get("userId").toString();
            String iStr = r.get("itemId").toString();
            if (!userMap.containsKey(uStr)) userMap.put(uStr, uCounter++);
            if (!itemMap.containsKey(iStr)) itemMap.put(iStr, iCounter++);
        }
        System.out.println("Ratings file: " + userMap.size() + " users, " + itemMap.size() + " items.");

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outFile))) {
            for (Map<String, Object> r : ratingsList) {
                String uStr = r.get("userId").toString();
                String iStr = r.get("itemId").toString();
                Object v = r.get("rating");
                int userOut = userMap.get(uStr) + 1;
                int itemOut = itemMap.get(iStr) + 1;
                bw.write(userOut + "\t" + itemOut + "\t" + v + "\n");
            }
        }
        return new IdMappings(userMap, itemMap);
    }

    public static void writeLibrecItemCategoryFile(File outFile,
                                                   List<Map<String, Object>> itemCategoryList,
                                                   Map<String, Integer> itemMap) throws IOException {
        Map<String, Integer> categoryMap = new HashMap<>();
        int cCounter = 0;

        for (Map<String, Object> itemCat : itemCategoryList) {
            Object cObj = itemCat.get("category");
            if (cObj == null) continue;
            String cStr = cObj.toString();
            if (!categoryMap.containsKey(cStr)) {
                categoryMap.put(cStr, cCounter++);
            }
        }
        System.out.println("Item category file: " + categoryMap.size() + " unique categories.");

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outFile))) {
            for (Map<String, Object> itemCat : itemCategoryList) {
                Object iObj = itemCat.get("itemId");
                Object cObj = itemCat.get("category");
                if (iObj == null || cObj == null) continue;

                String iStr = iObj.toString();
                String cStr = cObj.toString();

                if (itemMap.containsKey(iStr)) {
                    int itemOut = itemMap.get(iStr) + 1;
                    int catOut = categoryMap.get(cStr) + 1;
                    bw.write(itemOut + "\t" + catOut + "\t1.0\n");
                }
            }
        }
    }

    // === LibRec configuration ===

    static Configuration prepareCfConfiguration(Path tmpDir, Properties props) {
        Configuration conf = new Configuration();
        conf.set("dfs.data.dir", tmpDir.toString());
        conf.set("data.input.path", LIBREC_INPUT_FILENAME);
        conf.set("data.column.format", "UIR");
        conf.set("data.model.format", "text");
        
        // GBPR parameters (optimized for ranking)
        conf.setInt("rec.factor.number", 64);
        conf.setInt("rec.iterator.maximum", 100);
        conf.setDouble("rec.iterator.learnrate", 0.05);
        conf.setDouble("rec.user.regularization", 0.01);
        conf.setDouble("rec.item.regularization", 0.01);
        conf.setDouble("rec.group.regularization", 0.01);
        conf.setInt("rec.recommender.group.size", 5);
        
        // Rating scale
        conf.set("rec.recommender.rating.max", props.getProperty("rec.recommender.rating.max", "5.0"));
        conf.set("rec.recommender.rating.min", props.getProperty("rec.recommender.rating.min", "1.0"));
        
        // Ranking
        conf.setBoolean("rec.recommender.isranking", true);
        
        return conf;
    }

    static Configuration prepareCbConfiguration(Path tmpDir, Properties props) {
        Configuration conf = new Configuration();
        conf.set("dfs.data.dir", tmpDir.toString());
        conf.set("data.input.path", LIBREC_ITEM_CATEGORY_FILENAME);
        conf.set("data.column.format", "UIR");
        conf.set("rec.recommender.similarity.key", "user");
        conf.setBoolean("rec.recommender.isranking", Boolean.parseBoolean(props.getProperty("rec.isranking", "true")));
        conf.setInt("rec.similarity.shrinkage", Integer.parseInt(props.getProperty("rec.similarity.shrinkage", "10")));
        conf.set("rec.similarity.class", props.getProperty("rec.similarity.class", "net.librec.similarity.CosineSimilarity"));
        return conf;
    }

    // === Check contiguity ===

    static void checkLibrecFileContiguity(File file) throws IOException {
        Set<Integer> userIds = new HashSet<>();
        Set<Integer> itemIds = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 3) continue;
                int userId = Integer.parseInt(parts[0]);
                int itemId = Integer.parseInt(parts[1]);
                userIds.add(userId);
                itemIds.add(itemId);
            }
        }
    }

    // === Combine CF + CB ===

    public static RecommendedList combineRecommendations(RecommendedList cfList,
                                                         RecommendedList cbList,
                                                         double cfWeight,
                                                         DataModel dataModel) {
        double cbWeight = 1.0 - cfWeight;
        int numUsers = dataModel.getUserMappingData().size();

        if (cfWeight >= 1.0 || cbList == null || cbList.size() == 0) return cfList;
        if (cfWeight <= 0.0 || cfList == null || cfList.size() == 0) return cbList;

        RecommendedList hybridList = new RecommendedList(numUsers);

        for (int userIdx = 0; userIdx < numUsers; userIdx++) {
            List<KeyValue<Integer, Double>> cfRecs =
                    (cfList != null && userIdx < cfList.size())
                            ? cfList.getKeyValueListByContext(userIdx)
                            : null;
            List<KeyValue<Integer, Double>> cbRecs =
                    (cbList != null && userIdx < cbList.size())
                            ? cbList.getKeyValueListByContext(userIdx)
                            : null;

            if ((cfRecs == null || cfRecs.isEmpty())
                    && (cbRecs == null || cbRecs.isEmpty())) {
                hybridList.addList(new ArrayList<>());
                continue;
            }

            Map<Integer, Double> combinedScores = new HashMap<>();

            if (cfRecs != null) {
                for (KeyValue<Integer, Double> kv : cfRecs) {
                    combinedScores.put(kv.getKey(), kv.getValue() * cfWeight);
                }
            }
            if (cbRecs != null) {
                for (KeyValue<Integer, Double> kv : cbRecs) {
                    combinedScores.merge(kv.getKey(), kv.getValue() * cbWeight, Double::sum);
                }
            }

            ArrayList<KeyValue<Integer, Double>> hybridRecsForUser =
                    combinedScores.entrySet().stream()
                            .map(e -> new KeyValue<>(e.getKey(), e.getValue()))
                            .sorted((kv1, kv2) -> kv2.getValue().compareTo(kv1.getValue()))
                            .collect(Collectors.toCollection(ArrayList::new));

            hybridList.addList(hybridRecsForUser);
        }

        return hybridList;
    }

    // === Debug train/test ===

    private static void debugTrainTestSplit(DataModel dataModel) {
        SequentialAccessSparseMatrix train =
                (SequentialAccessSparseMatrix) dataModel.getTrainDataSet();
        SequentialAccessSparseMatrix test =
                (SequentialAccessSparseMatrix) dataModel.getTestDataSet();
        int numUsers = Math.min(train.rowSize(), test.rowSize());
        for (int u = 0; u < numUsers; u++) {
            int trainCount = train.row(u).size();
            int testCount = test.row(u).size();
            System.out.printf("User %d: train ratings=%d, test ratings=%d%n",
                    u, trainCount, testCount);
        }
    }

    // === Debug context ===

    private static void debugContext(RecommenderContext context,
                                     Recommender recommender,
                                     DataModel dataModel) {
        System.out.println("=== DEBUG START ===");
        System.out.println("Train set size: " + dataModel.getTrainDataSet().size());
        System.out.println("Test set size: " + dataModel.getTestDataSet().size());
        System.out.println("Recommender class: " + recommender.getClass().getName());
        System.out.println("Context params: " + context.getConf().toString());
        System.out.println("=== DEBUG END ===");
    }

    // === Evaluation ===

    private static void evaluateRecommendations(Configuration conf,
                                                Recommender recommender,
                                                DataModel dataModel,
                                                RecommenderContext context,
                                                RecommendedList recommendedList,
                                                Properties props) throws Exception {
        if (recommendedList == null || recommendedList.size() == 0) {
            System.err.println("No recommendations produced by LibRec.");
            return;
        }

        EvalContext evalContext = new EvalContext(
                conf,
                recommender,
                dataModel.getTestDataSet());

        evalContext.setRecommendedList(recommendedList);
        int topN = Integer.parseInt(props.getProperty("rec.topN", "10"));

        RecommenderEvaluator ndcgEvaluator = new NormalizedDCGEvaluator();
        ndcgEvaluator.setTopN(topN);
        lastNdcg = ndcgEvaluator.evaluate(evalContext);
        System.out.println("NDCG@10: " + lastNdcg);

        RecommenderEvaluator precisionEval = new PrecisionEvaluator();
        precisionEval.setTopN(topN);
        lastPrecision = precisionEval.evaluate(evalContext);
        System.out.println("Precision@10: " + lastPrecision);

        RecommenderEvaluator recallEval = new RecallEvaluator();
        recallEval.setTopN(topN);
        lastRecall = recallEval.evaluate(evalContext);
        System.out.println("Recall@10: " + lastRecall);

        lastF1 = (lastPrecision + lastRecall) > 0
                ? 2 * lastPrecision * lastRecall / (lastPrecision + lastRecall)
                : 0.0;
        System.out.println("F1@10: " + lastF1);

        lastDataModel = (TextDataModel) dataModel;
    }

    // === Save outputs ===

    private static String escapeXML(String text) {
        if (text == null) return "";
        return text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;");
    }

    public static void saveRecommendationsCSV(String filename,
                                              RecommendedList recs,
                                              TextDataModel dataModel,
                                              Map<String, Set<String>> itemCategories) throws IOException {
        BiMap<String, Integer> userMapping = dataModel.getUserMappingData();
        BiMap<String, Integer> itemMapping = dataModel.getItemMappingData();
        try (BufferedWriter csvW = new BufferedWriter(new FileWriter(filename))) {
            csvW.write("userId,itemId,score,categories\n");
            for (int userIdx = 0; userIdx < recs.size(); userIdx++) {
                List<KeyValue<Integer, Double>> items = recs.getKeyValueListByContext(userIdx);
                if (items == null) continue;
                String userId = userMapping.inverse().get(userIdx);
                if (userId == null) continue;

                for (KeyValue<Integer, Double> kv : items) {
                    String itemId = itemMapping.inverse().get(kv.getKey());
                    if (itemId == null) continue;

                    double score = kv.getValue();
                    Set<String> categories = itemCategories.getOrDefault(itemId, Collections.emptySet());
                    String categoryString = String.join("|", categories);

                    csvW.write(String.format("%s,%s,%.4f,\"%s\"%n",
                            userId, itemId, score, categoryString));
                }
            }
        }
    }

    public static void saveRecommendationsModel(String filename,
                                                RecommendedList recs,
                                                TextDataModel dataModel,
                                                Map<String, Set<String>> itemCategories) throws IOException {
        BiMap<String, Integer> userMapping = dataModel.getUserMappingData();
        BiMap<String, Integer> itemMapping = dataModel.getItemMappingData();
        try (BufferedWriter modelW = new BufferedWriter(new FileWriter(filename))) {
            modelW.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            modelW.write("<recommendations:UserItemMatrix xmi:version=\"2.0\" xmlns:xmi=\"http://www.omg.org/XMI\" xmlns:recommendations=\"http://org.rs.recommendations\" xmi:id=\"_BDihILMeEe-cHalY6VHacQ\">\n");
            for (int userIdx = 0; userIdx < recs.size(); userIdx++) {
                List<KeyValue<Integer, Double>> items = recs.getKeyValueListByContext(userIdx);
                if (items == null) continue;
                String userId = userMapping.inverse().get(userIdx);
                if (userId == null) continue;

                for (KeyValue<Integer, Double> kv : items) {
                    String itemId = itemMapping.inverse().get(kv.getKey());
                    if (itemId == null) continue;

                    double score = kv.getValue();
                    Set<String> categorySet = itemCategories.getOrDefault(itemId, Collections.emptySet());
                    String categories = String.join("|", categorySet);

                    modelW.write(String.format(
                            "  <rows _user=\"%s\" _item=\"%s\" value=\"%.4f\" _category=\"%s\" />%n",
                            escapeXML(userId),
                            escapeXML(itemId),
                            score,
                            escapeXML(categories)));
                }
            }
            modelW.write("</recommendations:UserItemMatrix>\n");
        }
    }
}
