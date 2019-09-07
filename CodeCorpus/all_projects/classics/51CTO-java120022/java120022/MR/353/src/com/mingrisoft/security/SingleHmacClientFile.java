/**
 * @jdk版本:1.6
 * @编码时间:2010-3-20
 */
package com.mingrisoft.security;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

import javax.crypto.Mac;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

/**
 * @author bwm
 * 
 */
public class SingleHmacClientFile {

	static String algorithm = "HmacMD5";

	static String keyFile = "keyData.dat";
	static String dataFile = "fileData.dat";

	/**
	 * HMAC加密
	 * 
	 * @param data
	 * @param key
	 * @return
	 * @throws NoSuchAlgorithmException
	 * @throws InvalidKeyException
	 */
	public byte[] encryptHMAC(byte[] data) throws NoSuchAlgorithmException,
			InvalidKeyException {
		byte key[] = readFile(keyFile);
		SecretKey secretKey = new SecretKeySpec(key, algorithm);
		Mac mac = Mac.getInstance(secretKey.getAlgorithm());
		mac.init(secretKey);
		return mac.doFinal();

	}
	
	/**
	 * 根据fileName读取数据文件
	 * 
	 * @param fileName
	 * @return
	 */
	public byte[] readFile(String fileName) {

		// 读取
		try {
			File file = new File(fileName);
			FileInputStream fileInputStream = new FileInputStream(file);
			byte[] data = new byte[(int) file.length()];
			fileInputStream.read(data);
			fileInputStream.close();
			return data;
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;

	}
}
