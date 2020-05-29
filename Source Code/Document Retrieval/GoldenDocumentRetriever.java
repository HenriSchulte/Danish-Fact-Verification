package dk.itu.feverda;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.analysis.da.DanishAnalyzer;

public class GoldenDocumentRetriever {
    public static void main(String[] args) throws Exception {
        // Wiki index
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        DanishAnalyzer analyzer = new DanishAnalyzer();
        Path indexPath = Paths.get("wikiIndex/");
        Directory directory;
        // Check if wiki index exists from previous build
        if (Files.exists(indexPath)) {
            System.out.println("Reusing previous wiki index.");
            directory = FSDirectory.open(indexPath);
        } else {
            System.out.println("Building wiki index...");
            Files.createDirectory(indexPath);
            directory = FSDirectory.open(indexPath);
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            IndexWriter iWriter = new IndexWriter(directory, config);

            // Build index from Wikipedia data
            String wikiPath = "dawiki-latest-pages-articles-parsed.tsv";
            Scanner wikiSc = new Scanner(classLoader.getResourceAsStream(wikiPath));
            wikiSc.nextLine(); // skip header
            List<String> invalidLines = new ArrayList<>();
            while(wikiSc.hasNextLine()) {
                String line = wikiSc.nextLine();
                // System.out.println(line);
                String[] splitline = line.split("\t");
                if (splitline.length == 4) {
                    String evidence = splitline[1];
                    String title = splitline[3];
                    addDoc(iWriter, evidence, title);
                } else {
                    invalidLines.add(line);
                }   
            }
            System.out.println("Unable to parse " + invalidLines.size() + " lines.");
            // for (String line : invalidLines) {
            //     System.out.println("INVALID: " + line);
            // }
            wikiSc.close();
            iWriter.close();
        }

        DirectoryReader iReader = DirectoryReader.open(directory);
        IndexSearcher iSearcher = new IndexSearcher(iReader);

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String outFileName = "out/retrieval_" + dtf.format(now) + ".jsonl";
        OutputStreamWriter outWriter = new OutputStreamWriter(new FileOutputStream(outFileName), StandardCharsets.UTF_8);
        // outWriter.append("claim\tevidence\tlabel\n");

        // Set up models for tokenizing and POS tagging
        InputStream tokenModelIn = classLoader.getResourceAsStream("da-token.bin");
        TokenizerModel tokenModel = new TokenizerModel(tokenModelIn);
        TokenizerME tokenizer = new TokenizerME(tokenModel);
        InputStream POSModelIn = classLoader.getResourceAsStream("da-pos-perceptron.bin");
        POSModel POSModel = new POSModel(POSModelIn);
        POSTaggerME POSTagger = new POSTaggerME(POSModel);

        if (args.length < 2) {
            // Read our annotations
            String testPath = args.length == 1 ? args[0] : "claims.tsv";
            System.out.println("Reading " + testPath + "...");
            Scanner testSc = new Scanner(new File(testPath));
            testSc.nextLine(); // skip header
            System.out.println("Selecting sentences through noun-clause queries...");
            while(testSc.hasNextLine()) {
                String[] line = testSc.nextLine().split("\t");
                String claim = line[1];
                String label = line[2];
                processLine(claim, label, tokenizer, iSearcher, analyzer, POSTagger, outWriter, classLoader, 5, 2);
            }
            testSc.close();
        } else if (args.length == 2) {
            System.out.println(String.format("Received claim '%s' with label '%s' from args.", args[0], args[1]));
            processLine(args[0], args[1], tokenizer, iSearcher, analyzer, POSTagger, outWriter, classLoader, 5 , 2);
        } else if (args.length == 3) {
            // Read our annotations
            String testPath = args[0];
            int k = Integer.parseInt(args[1]);
            int sentK = Integer.parseInt(args[2]);
            System.out.println("Reading " + testPath + "...");
            Scanner testSc = new Scanner(new File(testPath));
            testSc.nextLine(); // skip header
            System.out.println("Selecting sentences through noun-clause queries...");
            while(testSc.hasNextLine()) {
                String[] line = testSc.nextLine().split("\t");
                String claim = line[1];
                String label = line[2];
                processLine(claim, label, tokenizer, iSearcher, analyzer, POSTagger, outWriter, classLoader, k, sentK);
            }
            testSc.close();
        } else {
            System.out.println("Cannot parse arguments!");
        }
        
        // Cleanup
        outWriter.close();
        iReader.close();
        directory.close();
        // IOUtils.rm(indexPath);
        System.out.println("Done.");
    }

    private static void processLine(String claim, String label, TokenizerME tokenizer, IndexSearcher iSearcher, DanishAnalyzer analyzer, POSTaggerME POSTagger, OutputStreamWriter outWriter, ClassLoader classLoader, int k, int sentK) throws Exception { 
        String[] tokenizedClaim = tokenizer.tokenize(claim);
        String[] claimPOS = POSTagger.tag(tokenizedClaim);
        String nounClaim = "";
        for (int i = 0; i < tokenizedClaim.length; i++) {
            String token = tokenizedClaim[i];
            String tag = claimPOS[i];
            if (tag.startsWith("N")) nounClaim += " " + token;
        }
        String escapedClaim = QueryParser.escape(nounClaim.trim());

        // Searching the index
        QueryParser parser = new QueryParser("evidence", analyzer);
        Query query = parser.parse(escapedClaim);
        ScoreDoc[] hits = iSearcher.search(query, k).scoreDocs;

        // System.out.println(String.format("\nClaim: %s\nResults:", claim));
        List<String> evidence = new ArrayList<>();
        for (int i = 0; i < hits.length; i++) {
            Document hitDoc = iSearcher.doc(hits[i].doc);
            evidence.add(hitDoc.get("evidence"));
            // System.out.println(String.format("\t%d: %s", i + 1, hitDoc.get("evidence")));
        }

        String joinedEvidence = String.join(" ", evidence); // Concatenate to one string

        // Split into sentences
        InputStream modelIn = classLoader.getResourceAsStream("da-sent.bin");
        SentenceModel sentModel = new SentenceModel(modelIn);
        SentenceDetectorME sentenceDetector = new SentenceDetectorME(sentModel);
        String[] sentences = sentenceDetector.sentDetect(joinedEvidence);

        // Building the sentence index
        DanishAnalyzer sentAnalyzer = new DanishAnalyzer();
        Path sentIndexPath = Files.createTempDirectory("sentIndex");
        Directory sentDirectory = FSDirectory.open(sentIndexPath);
        IndexWriterConfig sentConfig = new IndexWriterConfig(sentAnalyzer);
        IndexWriter sentIWriter = new IndexWriter(sentDirectory, sentConfig);

        // System.out.println("\n" + claim);
        for(int i = 0; i < sentences.length; i++) {
            addSent(sentIWriter, sentences[i], i);
            // System.out.println(sentences[i]);
        }
        sentIWriter.close();

        // Searching the sent index
        DirectoryReader sentIReader = DirectoryReader.open(sentDirectory);
        IndexSearcher sentISearcher = new IndexSearcher(sentIReader);
        QueryParser sentParser = new QueryParser("sentence", sentAnalyzer);
        Query sentQuery = sentParser.parse(escapedClaim);
        ScoreDoc[] sentHits = sentISearcher.search(sentQuery, sentK).scoreDocs;

        List<String> selectedSentences = new ArrayList<>();
        for (int j = 0; j < sentHits.length; j++) {
            Document hitDoc = sentISearcher.doc(sentHits[j].doc);
            selectedSentences.add(hitDoc.get("sentence"));
        }

        String joinedSelectedEvidence = String.join(" ", selectedSentences); // Concatenate to one string
        // outWriter.append(claim).append("\t").append(joinedSelectedEvidence).append("\t").append(label).append("\n");
        outWriter.append("{\"claim\":\"").append(claim.replace("\"", "\\\""))
            .append("\",\"evidence\":\"").append(joinedSelectedEvidence.replace("\"", "\\\""))
            .append("\",\"label\":\"").append(label).append("\"}\n");

        sentIReader.close();
        sentDirectory.close();
        IOUtils.rm(sentIndexPath);
    }

    private static void addDoc(IndexWriter w, String evidence, String title) throws IOException {
        Document doc = new Document();
        doc.add(new Field("evidence", evidence, TextField.TYPE_STORED));
        doc.add(new Field("title", title, StringField.TYPE_STORED));
        w.addDocument(doc);
    }

    private static void addSent(IndexWriter w, String sentence, int index) throws IOException {
        Document doc = new Document();
        doc.add(new Field("sentence", sentence, TextField.TYPE_STORED));
        doc.add(new IntPoint("index", index));
        w.addDocument(doc);
    }
}