import java.io.FileInputStream;
import java.util.List;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.springframework.core.io.ClassPathResource;

public class Dummy{
	public static void main(String[] args) {
		try {
			String filePath = "/home/santhosh/workspaces/DeepLearning_SRL/"
					+ "dataset/wikipedia2text-extracted.txt";
			// Strip white space before and after for each line
	        SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);
	        // Split on white spaces in the line to get words
	        TokenizerFactory t = new DefaultTokenizerFactory();
	        t.setTokenPreProcessor(new CommonPreprocessor());
	        Tokenizer tok  = t.create(new FileInputStream(filePath));
	        while(tok.hasMoreTokens())
	        	System.out.println(tok.nextToken());
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
}
