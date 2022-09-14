/**
 *second 
 */
package com.mingrisoft.security;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.KeyAgreement;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.SecretKey;
import javax.crypto.interfaces.DHPublicKey;
import javax.crypto.spec.DHParameterSpec;

/**
 * @author baiweiming
 * 
 */
public class BothDHClientFile {

	private String keyAlgorithm = "DH";
	private String secretAlgorithm = "DES";
	// 服务端数据文件
	private String serverdataFile = "fileServerData.dat";
	// 客户端数据文件
	private String clientdataFile = "fileClientData.dat";
	// 服务端公钥文件
	private String publicServerkeyFile = "keyServerPublicData.dat";
	// 客户端公钥文件
	private String publicClientkeyFile = "keyClientPublicData.dat";
	// 客户端私钥文件
	private String privateClientkeyFile = "keyClientPrivateData.dat";

	/**
	 * 把数据写到指定的文件上
	 * 
	 * @param data
	 *            数据
	 * @param fileName
	 *            文件名称
	 */
	public void writeFile(byte[] data, String fileName) {
		try {
			FileOutputStream fileOutputStream = new FileOutputStream(fileName);
			fileOutputStream.write(data);
			fileOutputStream.close();
		} catch (FileNotFoundException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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

	/**
	 * 生成客户端密钥对
	 */
	public void generateClientKeyFile() {
		KeyPairGenerator keyPairGen = null;

		try {
			byte[] publicServerkey = readFile(publicServerkeyFile);
			X509EncodedKeySpec x509EncodedKeySpec = new X509EncodedKeySpec(
					publicServerkey);
			KeyFactory keyFactory = KeyFactory.getInstance(keyAlgorithm);
			PublicKey publicKey = keyFactory.generatePublic(x509EncodedKeySpec);
			DHParameterSpec dhParameterSpec = ((DHPublicKey) publicKey)
					.getParams();

			keyPairGen = KeyPairGenerator
					.getInstance(keyFactory.getAlgorithm());
			keyPairGen.initialize(dhParameterSpec);

		} catch (NoSuchAlgorithmException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidKeySpecException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidAlgorithmParameterException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		KeyPair keyPair = keyPairGen.generateKeyPair();

		// 公钥
		PublicKey publicKey = keyPair.getPublic();
		writeFile(publicKey.getEncoded(), publicClientkeyFile);

		// 私钥
		PrivateKey privateKey = keyPair.getPrivate();
		writeFile(privateKey.getEncoded(), privateClientkeyFile);
	}

	/**
	 * 生成客户端机密密钥
	 * 
	 * @return
	 */
	private SecretKey getClientSecretKey() {
		byte[] privateClientKey = readFile(privateClientkeyFile);
		byte[] publicServerKey = readFile(publicServerkeyFile);

		X509EncodedKeySpec x509KeySpec = new X509EncodedKeySpec(publicServerKey);
		PKCS8EncodedKeySpec pkcs8KeySpec = new PKCS8EncodedKeySpec(
				privateClientKey);

		PublicKey publicKey = null;
		KeyFactory keyFactory = null;
		Key privateKey = null;
		KeyAgreement keyAgree = null;

		try {
			keyFactory = KeyFactory.getInstance(keyAlgorithm);
			publicKey = keyFactory.generatePublic(x509KeySpec);
			// privateKey = keyFactory.generatePrivate(x509KeySpec);
			privateKey = keyFactory.generatePrivate(pkcs8KeySpec);
			// 创建密钥协议
			keyAgree = KeyAgreement.getInstance(keyFactory.getAlgorithm());
			keyAgree.init(privateKey);
			keyAgree.doPhase(publicKey, true);
			return keyAgree.generateSecret(secretAlgorithm);
		} catch (InvalidKeyException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalStateException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchAlgorithmException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidKeySpecException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * 客户端数据解密
	 * 
	 * @param data
	 * @param key
	 * @return
	 * @throws Exception
	 */
	public byte[] decryptForClient() {
		SecretKey secretKey = getClientSecretKey();

		try {
			byte[] data = readFile(serverdataFile);
			Cipher cipher = Cipher.getInstance(secretKey.getAlgorithm());
			cipher.init(Cipher.DECRYPT_MODE, secretKey);
			return cipher.doFinal(data);
		} catch (NoSuchAlgorithmException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchPaddingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidKeyException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalBlockSizeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (BadPaddingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * 客户端数据加密
	 * 
	 * @param data
	 */
	public void encryptForClient(byte[] data) {

		SecretKey secretKey = getClientSecretKey();

		try {
			Cipher cipher = Cipher.getInstance(secretKey.getAlgorithm());
			cipher.init(Cipher.ENCRYPT_MODE, secretKey);
			writeFile(cipher.doFinal(data), clientdataFile);
		} catch (NoSuchAlgorithmException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchPaddingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidKeyException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalBlockSizeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (BadPaddingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void main(String[] arg) {
		String data = "服务端你好，我是客户端";
		BothDHServerFile bothDHServerFile = new BothDHServerFile();
		BothDHClientFile bothDHClientFile = new BothDHClientFile();

		// 生成服务端密钥对
		bothDHServerFile.generateServerKeyFile();
		// 生成客户端密钥对
		bothDHClientFile.generateClientKeyFile();
		// 客户端加密
		bothDHClientFile.encryptForClient(data.getBytes());
		// 服务端解密
		byte[] data1 = bothDHServerFile.decryptForServer();
		System.out.println("Client原始数据：" + data);
		System.out.println("Servet解密数据：" + new String(data1));

	}

}
