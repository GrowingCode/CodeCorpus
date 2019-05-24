package environment;

import java.io.File;

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
		
		
	}
	
}
