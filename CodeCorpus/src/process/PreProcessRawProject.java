package process;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import util.ZIPUtil;

public class PreProcessRawProject {

//	public static final String user_home = System.getProperty("user.home");
//	public static final String unzip_dir = user_home + "/PPYYXUnzip";

	public static void PreProcessOneProject(File f) {
		assert f.getName().endsWith(".zip");
		try {
			ZIPUtil.Unzip(f, new File("all_projects"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void PreProcess() throws Exception {
		JsonParser parser = new JsonParser();
		JsonObject object = (JsonObject) parser.parse(new FileReader("config.json"));
		JsonArray array = object.get("preprocess").getAsJsonArray();
		ArrayList<File> wait_handles = new ArrayList<File>();
		if (array.size() == 0) {
			File raw_file = new File("raw_projects");
			wait_handles.addAll(Arrays.asList(raw_file.listFiles()));
		} else {
			int as = array.size();
			for (int a=0;a<as;a++) {
				JsonElement ele = array.get(a);
				String name = ele.getAsString();
				File f = new File("raw_projects" + "/" + name);
				wait_handles.add(f);
			}
		}
		for (File f : wait_handles) {
			PreProcessOneProject(f);
		}
	}

}
