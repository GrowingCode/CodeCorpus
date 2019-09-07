/**
 *second 
 */
package com.mingrisoft.DOM_demo;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

/**
 * @author baiweiming
 * 
 */
public class ParserInputStream {

	/**
	 * 使用InputStream读取XML文件
	 * 
	 * @param path
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public void parseInputStream(String path)
			throws ParserConfigurationException, SAXException, IOException {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = documentBuilderFactory
				.newDocumentBuilder();
		InputStream is = new FileInputStream(path);
		dombuilder.parse(is);
		is.close();

	}

	public static void main(String[] arg) {
		ParserInputStream parserFile = new ParserInputStream();
		String path = "xmldemo/books.xml";
		try {

			parserFile.parseInputStream(path);
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
