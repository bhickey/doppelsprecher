package doppelsprecher.nlp;

import static com.google.common.base.Preconditions.checkNotNull;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.TreeMap;

import kylm.model.ngram.NgramLM;
import kylm.model.ngram.smoother.MKNSmoother;
import kylm.model.ngram.smoother.NgramSmoother;
import kylm.reader.SentenceReader;
import kylm.reader.TextFileSentenceReader;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.inject.Inject;

public class Model {
	public static class Builder {
		private NgramSmoother smoother = new MKNSmoother();
		private Integer length;
		private String fileName;
		
		@Inject
		public Builder() {}
		
		public Builder setFileName(String fileName) {
			this.fileName = fileName;
			return this;
		}
		
		public Builder setLength(int length) {
			this.length = length;
			return this;
		}
		
		public Builder setSmoother(NgramSmoother smoother) {
			this.smoother = smoother;
			return this;
		}

		public NgramLM build() throws IOException {
			SentenceReader reader = new TextFileSentenceReader(checkNotNull(fileName));
			NgramLM model = new NgramLM(length, smoother);
			model.trainModel(reader);
			return model;
		}
	}

	private final TreeMap<Integer, NgramLM> models = Maps.newTreeMap();
	
	private Model(Builder builder) {
		checkNotNull(builder.length);
		try {
			SentenceReader reader = new TextFileSentenceReader(checkNotNull(builder.fileName));
			Iterator<String[]> iterator = reader.iterator();
			while (iterator.hasNext()) {
				String[] sentence = iterator.next();
				System.out.println(Arrays.toString(sentence));
			}
			for (int i = 0; i < builder.length; i++) {
				NgramLM model = new NgramLM(i, checkNotNull(builder.smoother));
				model.trainModel(reader);
				models.put(i + 1, model);
			}
		} catch (IOException e) {
			Throwables.propagate(e);
		}
	}

	public TreeMap<Integer, Iterable<Double>> entropy(String[] sentence) {
		TreeMap<Integer, Iterable<Double>> map = Maps.newTreeMap();
		for (Entry<Integer, NgramLM> entry : models.entrySet()) {
			ImmutableList.Builder<Double> builder = ImmutableList.builder();
			for (float entropy : entry.getValue().getWordEntropies(sentence)) {
				builder.add(Double.valueOf(entropy));
			}
			map.put(entry.getKey(), builder.build());
		}
		return map;
	}
}
