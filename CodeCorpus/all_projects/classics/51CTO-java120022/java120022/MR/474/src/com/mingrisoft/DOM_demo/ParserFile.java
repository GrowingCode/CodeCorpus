/**
 *second 
 */
package com.mingrisoft.DOM_demo;

import java.io.File;
import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

/**
 * @author baiweiming
 * 
 */
public class ParserFile {

	/**
	 * ¶ÁÈ¡XMLÎÄ¼þ
	 * 
	 * @param path
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public void parseReadFile(String path) throws ParserConfigurationException,
			SAXException, IOException {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = documentBuilderFactory
				.newDocumentBuilder();
		File file = new File(path);
		dombuilder.parse(file);

	}

	public static void main(String[] arg) {
		ParserFile parserFile = new ParserFile();
		String path = "xmldemo/books.xml";
		try {

			parserFile.parseReadFile(path);
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
