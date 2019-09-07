/**
 *second 
 */
package com.mingrisoft.DOM_demo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * @author baiweiming
 * 
 */
public class ElementValueDOMParserFile {

	private List<String> elementList = new ArrayList<String>();

	/**
	 * 读取XML文件
	 * 
	 * @param path
	 * @return
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public Document parseReadFile(String path)
			throws ParserConfigurationException, SAXException, IOException {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = documentBuilderFactory
				.newDocumentBuilder();
		File file = new File(path);
		return dombuilder.parse(file);
	}

	/**
	 * 获取{@link Node}的name和textcontent
	 * 
	 * @param parentNode
	 */
	public void getElementName(Node parentNode) {
		if (parentNode.hasChildNodes()) {
			NodeList nodeList = parentNode.getChildNodes();
			for (int i = 0; i < nodeList.getLength(); i++) {
				Node node = nodeList.item(i);
				if (node.hasChildNodes()) {
					getElementName(node);
					elementList.add(node.getNodeName() + "-"
							+ node.getTextContent());
				}
			}
		}
	}

	public List<String> getElementList() {
		return this.elementList;
	}

	public static void main(String[] arg) {
		ElementValueDOMParserFile parserFile = new ElementValueDOMParserFile();
		String path = "xmldemo/books.xml";
		try {

			Document document = parserFile.parseReadFile(path);
			parserFile.getElementName(document);
			List<String> list = parserFile.getElementList();
			System.out.println("XML元素名称和内容");
			System.out.println(list);
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
