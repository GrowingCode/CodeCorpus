package environment;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonIOException;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

import util.FileUtil;

public class EnvSetUp {
	
	public static final String user_home = System.getProperty("user.home");
	public static final String witness = user_home + "/YYXWitness";
	public static final String data = user_home + "/YYXData";
	
	public static void main(String[] args) {
		File witness_dir = new File(witness);
		if (witness_dir.exists()) {
			FileUtil.DeleteFile(witness_dir);
		}
		witness_dir.mkdirs();
		
		File data_dir = new File(data);
		if (data_dir.exists()) {
			FileUtil.DeleteFile(data_dir);
		}
		data_dir.mkdirs();
		
		JsonParser parser = new JsonParser();
		JsonObject object = null;
		try {
			object = (JsonObject) parser.parse(new FileReader("config.json"));
		} catch (JsonIOException | JsonSyntaxException | FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		{
			JsonArray array = object.get("witness").getAsJsonArray();
			ArrayList<File> wit_handles = new ArrayList<File>();
			int as = array.size();
			for (int a=0;a<as;a++) {
				JsonElement ele = array.get(a);
				String name = ele.getAsString();
				File f = new File("all_projects" + "/" + name);
				wit_handles.add(f);
			}
			for (File f : wit_handles) {
				try {
					FileUtil.CopyDirUnderSpecificDir(f.getAbsolutePath(), witness_dir.getAbsolutePath());
				} catch (IOException e) {
					e.printStackTrace();
					System.exit(1);
				}
			}
		}
		{
			JsonArray array = object.get("data").getAsJsonArray();
			ArrayList<File> data_handles = new ArrayList<File>();
			int as = array.size();
			for (int a=0;a<as;a++) {
				JsonElement ele = array.get(a);
				String name = ele.getAsString();
				File f = new File("all_projects" + "/" + name);
				data_handles.add(f);
			}
			for (File f : data_handles) {
				try {
					FileUtil.CopyDirUnderSpecificDir(f.getAbsolutePath(), data_dir.getAbsolutePath());
				} catch (IOException e) {
					e.printStackTrace();
					System.exit(1);
				}
			}
		}
		String ast_meta = user_home + "/AST_Metas";
		File dir = new File(ast_meta);
		if (!dir.exists()) {
			dir.mkdirs();
		}
		FileUtil.CopyFile(new File("config.json"), new File(ast_meta + "/" + "config.json"));
		System.out.println("Environment has been set up successfully.");
	}
	
}
